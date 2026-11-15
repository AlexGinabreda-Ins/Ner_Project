from pathlib import Path
import warnings

import pymupdf


def extract_text_from_pdf(path: str | Path) -> str:
    with pymupdf.open(path) as document:
        return "\n".join(page.get_text("text", sort=True) for page in document)


def extract_pdf_dataset(pdf_dir: str | Path, gold_samples: list[dict]) -> list[dict]:
    """Select PDFs by id, preserving gold order without using gold annotations."""
    directory = Path(pdf_dir)
    by_id = {}
    for path in sorted(directory.glob("*.pdf")):
        doc_id = path.stem.removesuffix("_scanned")
        if doc_id in by_id:
            raise ValueError(f"Duplicate PDF id: {doc_id}")
        by_id[doc_id] = path
    known_ids = {sample["id"] for sample in gold_samples}
    if set(by_id) - known_ids:
        raise ValueError("Some PDFs have no matching gold document")
    if not by_id:
        raise FileNotFoundError(f"No PDFs found in {directory}")
    samples = []
    for gold in gold_samples:
        if gold["id"] not in by_id:
            continue
        text = extract_text_from_pdf(by_id[gold["id"]])
        if not text.strip():
            warnings.warn(f"{gold['id']}: no text layer; OCR would be required")
        samples.append({"id": gold["id"], "language": gold.get("language"), "text": text})
    return samples
