from __future__ import annotations

import argparse
from pathlib import Path

from .agent import AgentConfig, MultimodalMemoryAgent
from .dataset import load_mmlongbench_doc_tsv
from .models import GenerationConfig, build_model
from .parallel import WorkerModelConfig, run_data_parallel
from .utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multimodal memory-agent flow on MMLongBench-DOC.")
    parser.add_argument("--tsv", required=True, help="Path to MMLongBench-DOC-style TSV metadata.")
    parser.add_argument("--image-root", required=True, help="Directory containing page images referenced by the TSV.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--backend", choices=["mock", "transformers", "vllm"], default="mock")
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dataset-name", default="MMLongBench_DOC")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=120,
        help="Maximum number of document pages visible to the agent, starting from page 1.",
    )
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print model inputs and outputs during inference.")

    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)

    parser.add_argument("--num-workers", type=int, default=1, help="Data-parallel worker count.")
    parser.add_argument(
        "--gpus",
        default="",
        help=(
            "GPU ids for workers. Use comma-separated ids for one GPU per worker, "
            "or semicolon-separated groups for tensor parallel workers, e.g. 0,1;2,3."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size per worker.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def parse_gpu_specs(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    if ";" in raw:
        return [group.strip() for group in raw.split(";") if group.strip()]
    return [gpu.strip() for gpu in raw.split(",") if gpu.strip()]


def main() -> None:
    args = parse_args()
    if args.max_pages < 1:
        raise ValueError("--max-pages must be at least 1.")
    output_path = Path(args.output)
    samples = load_mmlongbench_doc_tsv(
        args.tsv,
        args.image_root,
        max_pages=args.max_pages,
        allow_missing_images=args.allow_missing_images or args.backend == "mock",
    )

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
    )

    model_config = WorkerModelConfig(
        backend=args.backend,
        model_path=args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        generation_config=generation_config,
    )

    gpus = parse_gpu_specs(args.gpus)
    if args.num_workers > 1:
        if args.backend == "transformers":
            raise ValueError("Data-parallel multiprocessing is intended for backend=mock or backend=vllm.")
        if args.backend == "vllm" and gpus and len(gpus) < args.num_workers:
            raise ValueError("--gpus must provide at least one GPU id per worker when set.")
        run_data_parallel(
            samples=samples,
            output_path=output_path,
            model_config=model_config,
            dataset_name=args.dataset_name,
            max_pages=args.max_pages,
            verbose=args.verbose,
            num_workers=args.num_workers,
            gpus=gpus,
        )
    else:
        model = build_model(
            backend=args.backend,
            model_path=args.model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            generation_config=generation_config,
        )
        agent = MultimodalMemoryAgent(
            model,
            AgentConfig(dataset_name=args.dataset_name, max_pages=args.max_pages, verbose=args.verbose),
        )
        records = [agent.run_sample(sample) for sample in samples]
        write_jsonl(output_path, records)

    print(f"Wrote {len(samples)} records to {output_path}")


if __name__ == "__main__":
    main()
