# Multimodal MemAgent for MMLongBench-DOC

This directory contains a standalone migration of the MemAgent recurrent-memory idea to long-document visual understanding.
It does not modify `MemAgent` or `VLMEvalKit`.

The flow is:

1. Load an MMLongBench-DOC-style TSV file.
2. Resolve each sample's page images from a local image directory.
3. Iterate through pages one image at a time.
4. Ask a Qwen3-VL backend to update a compact memory after each page.
5. Ask for a final answer from the accumulated memory.
6. Write JSONL records with the full trace, the raw final response, a separate extracted final answer, and per-sample timing.

The default backend is `mock`, so the code can be developed and smoke-tested without downloading data or loading a model.

## Example: local mock smoke test

```bash
python -m mmemagent.run_mmlongbench_doc \
  --tsv examples/sample_mmlongbench_doc.tsv \
  --image-root examples/images \
  --output outputs/mock_results.jsonl \
  --backend mock \
  --limit 8 \
  --allow-missing-images \
  --verbose
```

## Example: Qwen3-VL with vLLM, data parallel over 8 GPUs

```bash
python -m mmemagent.run_mmlongbench_doc \
  --tsv /path/to/MMLongBench_DOC.tsv \
  --image-root /path/to/mmlongbench_doc_images \
  --output outputs/qwen3vl8b_memagent.jsonl \
  --backend vllm \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --num-workers 8 \
  --gpus 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 1 \
  --min-pixels 3136 \
  --max-pixels 200704 \
  --max-pages 120
```

With `--num-workers 8` and `--tensor-parallel-size 1`, the runner starts one process per GPU and partitions samples across workers. This is data parallel inference rather than vLLM tensor parallel inference.

For tensor parallel inference on a larger model, use fewer workers and increase `--tensor-parallel-size`, for example `--num-workers 1 --gpus 0,1,2,3,4,5,6,7 --tensor-parallel-size 8`.
For mixed data parallel plus tensor parallel, separate worker GPU groups with semicolons, for example `--num-workers 2 --gpus "0,1;2,3" --tensor-parallel-size 2`.

`--max-pages` limits the pages visible to the agent from page 1 onward. The default is `120`.
`--verbose` prints each model input in red and each model output in green, following the style used by VLMEvalKit.
`--limit N` runs only the first `N` samples, which is useful for smoke tests such as `--limit 8`.

## TSV expectations

The loader is intentionally permissive. It expects at least:

- `question`
- one of `image_path`, `images`, `page_images`, or `pages`

Image columns may be Python/JSON list strings such as `['doc_0_0.jpg', 'doc_0_1.jpg']`, JSON arrays, or comma-separated strings. Relative paths are resolved under `--image-root`.

Common MMLongBench metadata columns such as `index`, `doc_id`, `answer`, `answer_format`, `evidence_pages`, and `evidence_sources` are preserved in the output.
Large image-bearing columns such as `image`, `image_path`, `images`, `page_images`, and `pages` are used for loading but omitted from output metadata.

## Output fields

Each JSONL row includes:

- `sample_id`, `doc_id`, `question`, `metadata`
- `page_count`: number of pages actually visible to the agent after `--max-pages`
- `total_page_count`: number of page images listed by the TSV row
- `max_pages`: limit used for the run
- `page_outputs`: one entry per visible page with `page_number`, raw model `response`, and `seconds`
- `final_response`: raw model output from the answer step
- `final_answer`: extracted content from `<answer>...</answer>` when present, otherwise the raw final response
- `timing.total_seconds`: full sample inference time
