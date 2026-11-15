"""Run the required GLiNER experiments and save predictions and metrics."""
from pathlib import Path
import json
import platform
import re
import time

from ner.evaluation import ENTITY_LABELS, evaluate_ner_relaxed, evaluate_ner_strict
from ner.inference import DEFAULT_MODEL, LARGE_MODEL, load_gliner_model, predict_dataset
from ner.pdf import extract_pdf_dataset
from ner.utils import load_jsonl, save_responses_to_jsonl

ROOT = Path(__file__).resolve().parents[1]


def evaluate_experiment(samples, predictions, output_path, **metadata):
    """Align gold explicitly so a PDF subset never gets scored by position alone."""
    by_id = {sample["id"]: sample for sample in samples}
    gold = [by_id[prediction["id"]] for prediction in predictions]
    gold_path = output_path.with_name(output_path.stem + "_gold.jsonl")
    save_responses_to_jsonl(predictions, output_path)
    save_responses_to_jsonl(gold, gold_path)
    result = {
        **metadata, "documents": len(predictions),
        "strict": evaluate_ner_strict(gold_path, output_path, False),
        "relaxed": evaluate_ner_relaxed(gold_path, output_path, False),
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    import torch
    import pymupdf
    from importlib.metadata import version

    gold = load_jsonl(ROOT / "data/plain/validation.jsonl")
    inputs = {"plain": gold,
              "pdf_clean": extract_pdf_dataset(ROOT / "data/pdf_clean", gold),
              "pdf_scanned": extract_pdf_dataset(ROOT / "data/pdf_scanned", gold)}
    configuration = {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "torch": torch.__version__,
        "gliner": version("gliner"), "pymupdf": pymupdf.VersionBind,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "threads": 4, "threshold": 0.5, "window_tokens": 200, "overlap_tokens": 40,
        "label_descriptions": False,
        "inputs": {name: {"documents": len(rows),
                           "empty_texts": sum(not r["text"].strip() for r in rows)}
                   for name, rows in inputs.items()},
    }
    plain_by_id = {sample["id"]: sample["text"] for sample in gold}
    clean = inputs["pdf_clean"]
    configuration["clean_pdf_alignment"] = {
        "identical_texts": sum(s["text"] == plain_by_id[s["id"]] for s in clean),
        "equal_character_counts": sum(len(s["text"]) == len(plain_by_id[s["id"]]) for s in clean),
        "whitespace_normalized_equal": sum(
            re.sub(r"\s+", " ", s["text"]).strip()
            == re.sub(r"\s+", " ", plain_by_id[s["id"]]).strip() for s in clean),
    }
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    for model_name, short_name in [(DEFAULT_MODEL, "small"), (LARGE_MODEL, "large")]:
        print(f"Loading {model_name}", flush=True)
        model = load_gliner_model(model_name)
        for name, samples in inputs.items():
            if short_name == "large" and name != "plain":
                continue
            output = results_dir / name / f"gliner_{short_name}.jsonl"
            start = time.perf_counter()
            predictions = []
            for index, sample in enumerate(samples, 1):
                predictions.extend(predict_dataset(model, [sample], list(ENTITY_LABELS)))
                print(f"{short_name} {name}: {index}/{len(samples)}", flush=True)
            evaluate_experiment(gold, predictions, output, input=name, model=model_name,
                                label_descriptions=False, extraction="PyMuPDF" if name != "plain" else None,
                                seconds=round(time.perf_counter() - start, 3))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
