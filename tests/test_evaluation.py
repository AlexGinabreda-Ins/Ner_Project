import tempfile
import unittest
from pathlib import Path

from ner.evaluation import evaluate_ner_relaxed, evaluate_ner_strict
from ner.utils import save_responses_to_jsonl


class EvaluationTests(unittest.TestCase):
    def test_offsets_whitespace_and_repeated_mentions(self):
        gold = [{"id": "a", "text": "Ana Ana", "entities": [
            {"start": 0, "end": 3, "label": "PERSON_NAME", "text": "Ana"},
            {"start": 4, "end": 7, "label": "PERSON_NAME", "text": "Ana"},
        ]}]
        predictions = [{"id": "a", "text": " Ana ", "entities": [
            {"start": 1, "end": 4, "label": "PERSON_NAME", "text": " Ana "},
        ]}]
        with tempfile.TemporaryDirectory() as directory:
            g, p = Path(directory) / "gold.jsonl", Path(directory) / "pred.jsonl"
            save_responses_to_jsonl(gold, g)
            save_responses_to_jsonl(predictions, p)
            strict = evaluate_ner_strict(g, p, False)["micro"]
            relaxed = evaluate_ner_relaxed(g, p, False)
        self.assertEqual(strict["true_positives"], 0)
        self.assertEqual(strict["false_positives"], 1)
        self.assertEqual(strict["false_negatives"], 2)
        self.assertEqual(relaxed["occurrences"]["micro"]["recall"], 0.5)
        self.assertEqual(relaxed["unique"]["micro"]["f1"], 1.0)

    def test_misaligned_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            g, p = Path(directory) / "gold.jsonl", Path(directory) / "pred.jsonl"
            save_responses_to_jsonl([{"id": "a", "entities": []}], g)
            save_responses_to_jsonl([{"id": "b", "entities": []}], p)
            with self.assertRaises(ValueError):
                evaluate_ner_strict(g, p, False)
