from __future__ import annotations

import ast
import csv
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


def parse_listish(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and str(value) == "nan":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]

    text = str(value).strip()
    if not text:
        return []

    if text[0] in "[(":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except Exception:
                pass

    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def resolve_image_paths(values: Iterable[str], image_root: Path) -> list[str]:
    paths = []
    for value in values:
        path = Path(value)
        if not path.is_absolute():
            path = image_root / path
        paths.append(str(path))
    return paths


def read_tsv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


@contextmanager
def timer() -> Iterable[dict[str, float]]:
    box: dict[str, float] = {"start": time.perf_counter()}
    try:
        yield box
    finally:
        box["end"] = time.perf_counter()
        box["seconds"] = box["end"] - box["start"]

