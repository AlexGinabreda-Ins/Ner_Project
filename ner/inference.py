from __future__ import annotations

import re
from typing import Any

DEFAULT_MODEL = "gliner-community/gliner_small-v2.5"
LARGE_MODEL = "gliner-community/gliner_large-v2.5"


def load_gliner_model(model_name: str = DEFAULT_MODEL, device: str | None = None):
    import torch
    from gliner import GLiNER

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = GLiNER.from_pretrained(model_name).to(device)
    model.eval()
    return model


def split_into_chunks(text: str, size: int = 200, overlap: int = 40):
    """Yield overlapping windows with their original character offsets."""
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    tokens = list(re.finditer(r"\w+|[^\w\s]", text))
    for first in range(0, len(tokens), size - overlap):
        last = min(first + size, len(tokens))
        start, end = tokens[first].start(), tokens[last - 1].end()
        yield start, text[start:end]
        if last == len(tokens):
            break


def predict_entities(model, text: str, labels: list[str], threshold: float = 0.5):
    """Restore global offsets and deduplicate entities in overlapping windows."""
    found = {}
    for offset, chunk in split_into_chunks(text):
        for entity in model.predict_entities(chunk, labels, threshold=threshold):
            start, end = offset + int(entity["start"]), offset + int(entity["end"])
            key = (start, end, entity["label"])
            prediction = {"start": start, "end": end, "text": text[start:end],
                          "label": entity["label"], "score": float(entity["score"])}
            if key not in found or prediction["score"] > found[key]["score"]:
                found[key] = prediction
    return sorted(found.values(), key=lambda e: (e["start"], e["end"], e["label"]))


def predict_dataset(model, samples: list[dict[str, Any]], labels: list[str],
                    threshold: float = 0.5):
    return [{"id": sample["id"], "language": sample.get("language"),
             "text": sample["text"],
             "entities": predict_entities(model, sample["text"], labels, threshold)}
            for sample in samples]
