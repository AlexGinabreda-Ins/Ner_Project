import unittest
from pathlib import Path
import tempfile

import pymupdf

from ner.inference import predict_entities, split_into_chunks
from ner.llm import align_chat_response
from ner.pdf import extract_pdf_dataset, extract_text_from_pdf


class FakeModel:
    def predict_entities(self, text, labels, threshold):
        import re
        return [{"start": m.start(), "end": m.end(), "label": labels[0], "score": 0.9}
                for m in re.finditer("Ada Example", text)]


class PipelineTests(unittest.TestCase):
    def test_overlapping_windows_keep_global_spans_without_duplicates(self):
        text = "prefix " * 90 + "Ada Example " + "middle " * 90 + "Ada Example " + "tail " * 250
        entities = predict_entities(FakeModel(), text, ["PERSON_NAME"])
        self.assertEqual(len(entities), 2)
        for entity in entities:
            self.assertEqual(text[entity["start"]:entity["end"]], "Ada Example")
        self.assertEqual(predict_entities(FakeModel(), "", ["PERSON_NAME"]), [])
        with self.assertRaises(ValueError):
            list(split_into_chunks(text, size=4, overlap=4))

    def test_chat_alignment_counts_repeated_mentions(self):
        samples = [{"id": "x", "text": "Ada Example met Ada Example."}]
        response = [{"id": "x", "entities": [{"text": "Ada Example", "label": "PERSON_NAME"}]}]
        result = align_chat_response(samples, response)[0]["entities"]
        self.assertEqual([(e["start"], e["end"]) for e in result], [(0, 11), (16, 27)])

    def test_pdf_ids_follow_gold_order_and_missing_text_stays_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with pymupdf.open() as document:
                document.new_page().insert_text((72, 72), "Example document")
                document.save(directory / "b.pdf")
            with pymupdf.open() as document:
                document.new_page()
                document.save(directory / "a_scanned.pdf")
            self.assertIn("Example document", extract_text_from_pdf(directory / "b.pdf"))
            with self.assertWarns(UserWarning):
                rows = extract_pdf_dataset(directory, [{"id": "a"}, {"id": "b"}])
            self.assertEqual([row["id"] for row in rows], ["a", "b"])
            self.assertEqual(rows[0]["text"], "")
