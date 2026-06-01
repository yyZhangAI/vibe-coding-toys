from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import AgentConfig, MultimodalMemoryAgent
from .dataset import DocSample
from .models import GenerationConfig, build_model
from .utils import read_jsonl, write_jsonl


@dataclass
class WorkerModelConfig:
    backend: str
    model_path: str
    min_pixels: int | None
    max_pixels: int | None
    tensor_parallel_size: int
    gpu_memory_utilization: float
    generation_config: GenerationConfig


def split_samples(samples: list[DocSample], num_shards: int) -> list[list[DocSample]]:
    shards = [[] for _ in range(num_shards)]
    for idx, sample in enumerate(samples):
        shards[idx % num_shards].append(sample)
    return shards


def _worker_main(
    worker_id: int,
    samples: list[DocSample],
    output_path: str,
    model_config: WorkerModelConfig,
    dataset_name: str,
    max_pages: int,
    verbose: bool,
    gpu_spec: str | None,
) -> None:
    if gpu_spec:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_spec
    model = build_model(
        backend=model_config.backend,
        model_path=model_config.model_path,
        min_pixels=model_config.min_pixels,
        max_pixels=model_config.max_pixels,
        tensor_parallel_size=model_config.tensor_parallel_size,
        gpu_memory_utilization=model_config.gpu_memory_utilization,
        generation_config=model_config.generation_config,
    )
    agent = MultimodalMemoryAgent(
        model,
        AgentConfig(dataset_name=dataset_name, max_pages=max_pages, verbose=verbose),
    )
    records = []
    for sample in samples:
        record = agent.run_sample(sample)
        record["worker_id"] = worker_id
        record["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        records.append(record)
    write_jsonl(Path(output_path), records)


def run_data_parallel(
    samples: list[DocSample],
    output_path: Path,
    model_config: WorkerModelConfig,
    dataset_name: str,
    max_pages: int,
    verbose: bool,
    num_workers: int,
    gpus: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_path.parent / f".{output_path.stem}_shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shards = split_samples(samples, num_workers)
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    shard_paths: list[Path] = []

    for worker_id, shard in enumerate(shards):
        shard_path = tmp_dir / f"worker_{worker_id}.jsonl"
        shard_paths.append(shard_path)
        gpu_spec = gpus[worker_id] if worker_id < len(gpus) else None
        p = ctx.Process(
            target=_worker_main,
            args=(worker_id, shard, str(shard_path), model_config, dataset_name, max_pages, verbose, gpu_spec),
        )
        p.start()
        processes.append(p)

    failures = []
    for worker_id, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            failures.append((worker_id, p.exitcode))
    if failures:
        raise RuntimeError(f"Worker failures: {failures}")

    by_id: dict[str, dict[str, Any]] = {}
    for shard_path in shard_paths:
        if shard_path.exists():
            for record in read_jsonl(shard_path):
                by_id[str(record["sample_id"])] = record

    ordered = [by_id[sample.sample_id] for sample in samples if sample.sample_id in by_id]
    write_jsonl(output_path, ordered)
