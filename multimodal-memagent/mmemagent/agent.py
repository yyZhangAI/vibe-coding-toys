from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .dataset import DocSample
from .models import VisionLanguageModel
from .prompts import NO_MEMORY, build_final_prompt, build_page_prompt
from .utils import extract_answer, timer


@dataclass
class AgentConfig:
    dataset_name: str = "MMLongBench_DOC"
    max_pages: int = 120
    verbose: bool = False


class MultimodalMemoryAgent:
    def __init__(self, model: VisionLanguageModel, config: AgentConfig | None = None) -> None:
        self.model = model
        self.config = config or AgentConfig()

    def _log_input(self, title: str, image: str | None, prompt: str) -> None:
        if not self.config.verbose:
            return
        content: list[dict[str, str]] = []
        if image is not None:
            content.append({"type": "image", "value": image})
        content.append({"type": "text", "value": prompt})
        message = {"role": "user", "content": content}
        print(f"\n{'=' * 30} {title} {'=' * 30}", flush=True)
        print("\033[31m" + json.dumps(message, ensure_ascii=False, indent=2) + "\033[0m", flush=True)

    def _log_output(self, title: str, response: str) -> None:
        if not self.config.verbose:
            return
        print(f"{'-' * 30} {title} {'-' * 30}", flush=True)
        print("\033[32m" + response + "\033[0m", flush=True)

    def run_sample(self, sample: DocSample) -> dict[str, Any]:
        if self.config.max_pages < 1:
            raise ValueError("AgentConfig.max_pages must be at least 1.")
        memory = NO_MEMORY
        page_traces: list[dict[str, Any]] = []
        page_images = sample.page_images[: self.config.max_pages]

        with timer() as sample_timer:
            page_count = len(page_images)
            for page_idx, image_path in enumerate(page_images, start=1):
                prompt = build_page_prompt(
                    question=sample.question,
                    memory=memory,
                    page_number=page_idx,
                    page_count=page_count,
                )
                self._log_input(f"sample={sample.sample_id} page={page_idx}/{page_count} input", image_path, prompt)
                with timer() as turn_timer:
                    response = self.model.generate(image_path, prompt, dataset=self.config.dataset_name)
                self._log_output(f"sample={sample.sample_id} page={page_idx}/{page_count} output", response)
                memory = response.strip() or memory
                page_traces.append(
                    {
                        "page_number": page_idx,
                        "image": image_path,
                        "prompt": prompt,
                        "response": response,
                        "memory_after": memory,
                        "timing": {"seconds": turn_timer["seconds"]},
                    }
                )

            final_prompt = build_final_prompt(sample.question, memory)
            self._log_input(f"sample={sample.sample_id} final input", None, final_prompt)
            with timer() as final_timer:
                final_response = self.model.generate(None, final_prompt, dataset=self.config.dataset_name)
            self._log_output(f"sample={sample.sample_id} final output", final_response)
            final_answer = extract_answer(final_response)

        return {
            "sample_id": sample.sample_id,
            "doc_id": sample.doc_id,
            "question": sample.question,
            "page_count": len(page_images),
            "total_page_count": len(sample.page_images),
            "max_pages": self.config.max_pages,
            "metadata": sample.metadata,
            "page_traces": page_traces,
            "memory": memory,
            "final_prompt": final_prompt,
            "final_response": final_response,
            "final_answer": final_answer,
            "timing": {
                "total_seconds": sample_timer["seconds"],
                "final_step_seconds": final_timer["seconds"],
            },
        }
