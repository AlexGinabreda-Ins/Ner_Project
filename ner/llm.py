"""Convert a chat's entity-text response into the common prediction format."""
import re
from ner.evaluation import ENTITY_LABELS
from ner.experiments import ROOT, evaluate_experiment
from ner.utils import load_jsonl


def align_chat_response(samples: list[dict], responses: list[dict]) -> list[dict]:
    """Locate all exact mentions of the values selected by the chat, without gold spans."""
    by_id = {sample["id"]: sample for sample in samples}
    predictions = []
    if len({response["id"] for response in responses}) != len(responses):
        raise ValueError("Duplicate chat response id")
    for response in responses:
        sample = by_id[response["id"]]
        text = sample["text"]
        entities = {}
        for entity in response["entities"]:
            value, label = entity["text"], entity["label"]
            if label not in ENTITY_LABELS or not value:
                raise ValueError("Invalid entity label or empty value")
            matches = list(re.finditer(re.escape(value), text))
            if not matches:
                raise ValueError(f"Chat returned a value absent from {sample['id']}")
            for match in matches:
                key = (match.start(), match.end(), label)
                entities[key] = {"start": match.start(), "end": match.end(),
                                 "text": match.group(), "label": label}
        predictions.append({"id": sample["id"], "language": sample.get("language"),
                            "text": text,
                            "entities": sorted(entities.values(), key=lambda e: e["start"])})
    return predictions


def main():
    gold = load_jsonl(ROOT / "data/plain/validation_small.jsonl")
    responses = load_jsonl(ROOT / "results/llm/chat_response.jsonl")
    predictions = align_chat_response(gold, responses)
    evaluate_experiment(gold, predictions, ROOT / "results/llm/chat.jsonl",
                        input="plain (first two documents)", model="Codex chat",
                        label_descriptions=False, extraction=None,
                        method="Chat entity texts; deterministic exact-occurrence span alignment; no API")


if __name__ == "__main__":
    main()
