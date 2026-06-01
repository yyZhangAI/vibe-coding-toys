from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import parse_listish, read_tsv_rows, resolve_image_paths


IMAGE_COLUMNS = ("image_path", "images", "page_images", "pages")


@dataclass
class DocSample:
    sample_id: str
    question: str
    page_images: list[str]
    metadata: dict[str, Any]

    @property
    def doc_id(self) -> str | None:
        value = self.metadata.get("doc_id")
        return None if value is None else str(value)


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def load_mmlongbench_doc_tsv(
    tsv_path: str | Path,
    image_root: str | Path,
    max_pages: int | None = 120,
    allow_missing_images: bool = False,
) -> list[DocSample]:
    tsv_path = Path(tsv_path)
    image_root = Path(image_root)
    rows = read_tsv_rows(tsv_path)
    samples: list[DocSample] = []

    for offset, row in enumerate(rows):
        question = row.get("question")
        if not question:
            raise ValueError(f"Row {offset} is missing required column `question`.")

        image_value = _first_present(row, IMAGE_COLUMNS)
        image_names = parse_listish(image_value)
        page_images = resolve_image_paths(image_names, image_root)

        if not page_images:
            raise ValueError(f"Row {offset} has no page images in columns {IMAGE_COLUMNS}.")
        if not allow_missing_images:
            checked_images = page_images[:max_pages] if max_pages is not None else page_images
            missing = [p for p in checked_images if not Path(p).exists()]
            if missing:
                preview = ", ".join(missing[:3])
                raise FileNotFoundError(
                    f"Row {offset} references missing image files: {preview}. "
                    "Use --allow-missing-images for mock-only smoke tests."
                )

        sample_id = str(row.get("index") or row.get("id") or offset)
        samples.append(DocSample(sample_id=sample_id, question=question, page_images=page_images, metadata=row))

    return samples
