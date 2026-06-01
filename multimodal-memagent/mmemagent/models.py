from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


class VisionLanguageModel(Protocol):
    def generate(self, image: str | None, prompt: str, *, dataset: str | None = None) -> str:
        ...


@dataclass
class GenerationConfig:
    max_new_tokens: int = 2048
    temperature: float = 0.01
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0
    presence_penalty: float = 1.5


class MockVisionLanguageModel:
    def __init__(self, model_path: str = "mock-qwen3-vl", **_: Any) -> None:
        self.model_path = model_path
        self.turn = 0

    def generate(self, image: str | None, prompt: str, *, dataset: str | None = None) -> str:
        self.turn += 1
        if image is None:
            return (
                "Rationale: This is a mock final answer produced without loading a model. "
                "<answer>mock answer</answer>"
            )
        image_name = os.path.basename(image)
        return f"Mock memory update {self.turn}: inspected {image_name}; retain any relevant evidence for the question."


def _content_from_image_and_prompt(
    image: str | None,
    prompt: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image is not None:
        image_item: dict[str, Any] = {"type": "image", "value": image}
        if min_pixels is not None:
            image_item["min_pixels"] = min_pixels
        if max_pixels is not None:
            image_item["max_pixels"] = max_pixels
        content.append(image_item)
    content.append({"type": "text", "value": prompt})
    return content


class TransformersQwen3VLModel:
    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        generation_config: GenerationConfig | None = None,
        **_: Any,
    ) -> None:
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generation_config = generation_config or GenerationConfig()

        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

    def _prepare_messages(self, image: str | None, prompt: str) -> list[dict[str, Any]]:
        content = []
        for item in _content_from_image_and_prompt(image, prompt, self.min_pixels, self.max_pixels):
            if item["type"] == "image":
                image_payload: dict[str, Any] = {"type": "image", "image": item["value"]}
                if "min_pixels" in item:
                    image_payload["min_pixels"] = item["min_pixels"]
                if "max_pixels" in item:
                    image_payload["max_pixels"] = item["max_pixels"]
                content.append(image_payload)
            else:
                content.append({"type": "text", "text": item["value"]})
        return [{"role": "user", "content": content}]

    def generate(self, image: str | None, prompt: str, *, dataset: str | None = None) -> str:
        messages = self._prepare_messages(image, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = self.process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, "dtype"):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_config.max_new_tokens,
            do_sample=self.generation_config.temperature > 0,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            repetition_penalty=self.generation_config.repetition_penalty,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class VLLMQwen3VLModel:
    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        generation_config: GenerationConfig | None = None,
        limit_mm_per_prompt: int = 1,
        **_: Any,
    ) -> None:
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generation_config = generation_config or GenerationConfig()

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.process_vision_info = process_vision_info
        self.SamplingParams = SamplingParams
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": limit_mm_per_prompt},
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            seed=0,
        )

    def _prepare_messages(self, image: str | None, prompt: str) -> list[dict[str, Any]]:
        content = []
        for item in _content_from_image_and_prompt(image, prompt, self.min_pixels, self.max_pixels):
            if item["type"] == "image":
                image_payload: dict[str, Any] = {"type": "image", "image": item["value"]}
                if "min_pixels" in item:
                    image_payload["min_pixels"] = item["min_pixels"]
                if "max_pixels" in item:
                    image_payload["max_pixels"] = item["max_pixels"]
                content.append(image_payload)
            else:
                content.append({"type": "text", "text": item["value"]})
        return [{"role": "user", "content": content}]

    def generate(self, image: str | None, prompt: str, *, dataset: str | None = None) -> str:
        messages = self._prepare_messages(image, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        request: dict[str, Any] = {"prompt": text}
        mm_data: dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        if mm_data:
            request["multi_modal_data"] = mm_data
        if video_kwargs is not None:
            request["mm_processor_kwargs"] = video_kwargs

        params = self.SamplingParams(
            temperature=self.generation_config.temperature,
            max_tokens=self.generation_config.max_new_tokens,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            repetition_penalty=self.generation_config.repetition_penalty,
            presence_penalty=self.generation_config.presence_penalty,
        )
        outputs = self.llm.generate([request], sampling_params=params)
        return outputs[0].outputs[0].text


def build_model(
    backend: str,
    model_path: str,
    min_pixels: int | None,
    max_pixels: int | None,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    generation_config: GenerationConfig,
) -> VisionLanguageModel:
    if backend == "mock":
        return MockVisionLanguageModel(model_path=model_path)
    if backend == "transformers":
        return TransformersQwen3VLModel(
            model_path=model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            generation_config=generation_config,
        )
    if backend == "vllm":
        return VLLMQwen3VLModel(
            model_path=model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            generation_config=generation_config,
        )
    raise ValueError(f"Unsupported backend: {backend}")

