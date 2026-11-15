# NER Project

## Setup and execution

Use Python 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/). The environment was created with `uv add gliner pymupdf` and `uv add --dev ipykernel nbclient nbformat`. The resolved versions are recorded in `uv.lock`.

```sh
uv sync
uv run python -m unittest discover -s tests
uv run python -m ner.experiments
uv run python -m ner.llm
```

Open the project folder in PyCharm, select its `.venv` interpreter, and run the notebooks in numeric order. Run commands from the repository root. The first GLiNER run downloads the model weights. `ner.experiments` runs small GLiNER on plain text and both PDF inputs, then large GLiNER on plain text; predictions and aggregate metrics are saved under `results/`.

The local dataset contains 50 JSONL records, 50 clean PDFs and 10 scanned PDFs. The data and full predictions are deliberately excluded from GitHub at the owner's request. On another computer, obtain the instructor's files and place them at:

```text
data/plain/validation.jsonl
data/plain/validation_small.jsonl  (first two validation records)
data/pdf_clean/test-0001.pdf ... test-0050.pdf
data/pdf_scanned/test-0001_scanned.pdf ... test-0010_scanned.pdf
```

The chat experiment also needs the locally saved `results/llm/chat_response.jsonl`. A fresh clone needs a new response generated in a chat, with `id` and `entities` containing unique literal `text` and `label` values. See [the extraction protocol](results/llm/prompt.md); no API is used. The supplied local response can be evaluated with `ner.llm`.

## Short report

### Group members and organization

Alex Ginabreda. Individual project organized in three stages: environment and evaluation, implementation of the pipelines, and experiments with analysis. The assignment's evaluation helpers were reused. The reference project was consulted for its dataset and structure; its saved predictions were not used as experimental results.

### System and configuration

Plain text is loaded from JSONL. PDF text is extracted with PyMuPDF using `page.get_text("text", sort=True)` and pages are joined with a newline. Each PDF is matched by document id to its annotation record, with `_scanned` removed from scanned filenames. Missing PDFs are not replaced with the gold text.

The models are `gliner-community/gliner_small-v2.5` and `gliner-community/gliner_large-v2.5`. Small is used for all three input formats; large provides a second plain-text baseline. The seven original entity labels are passed directly, without label descriptions. The threshold is 0.5. Long texts use overlapping windows of 200 regex tokens, with an overlap of 40, to avoid silently losing the end of a document. Local offsets are restored to document coordinates, and identical predictions in overlapping windows are deduplicated by their highest score.

Strict evaluation matches `(start, end, label)`. A span starts at `start` and ends before `end`, so `text[start:end]` is the entity mention. Relaxed occurrences matches label and whitespace-normalized text while counting repeated mentions. Relaxed unique counts each identical label/text pair once per document. All evaluations include micro and per-entity precision, recall and F1.

The chat experiment uses the first two documents, one Spanish and one Catalan, in the current Codex chat. The chat selected literal entity values; `ner.llm` located all exact occurrences and computed offsets without using gold entity annotations. It is therefore a chat-plus-alignment experiment, rather than unaided numeric span generation. Only the same two documents are used for the matched GLiNER comparison. The chat's exact underlying model identifier is not exposed in this project.

The measured run used Windows 10, Python 3.13.18, GLiNER 0.2.30, PyMuPDF 2.1 and PyTorch 2.15.0+cpu. Inference ran on the CPU with four threads. Full configuration and dataset coverage are recorded in `results/configuration.json`.

### Results summary

All values below are micro scores on a 0-1 scale. No predictions from the reference project were reused. Detailed per-entity scores are in the published `results/**/*.metrics.json` files; raw predictions stay local.

| Input | Model | N | Label descriptions | Extraction | Evaluation | P | R | F1 |
|---|---|---:|---|---|---|---:|---:|---:|
| Plain text | small | 50 | No | - | Strict | 0.708 | 0.849 | 0.772 |
| Plain text | small | 50 | No | - | Relaxed occurrences | 0.708 | 0.849 | 0.772 |
| Plain text | small | 50 | No | - | Relaxed unique | 0.672 | 0.878 | 0.761 |
| Plain text | large | 50 | No | - | Strict | 0.928 | 0.912 | 0.920 |
| Plain text | large | 50 | No | - | Relaxed occurrences | 0.928 | 0.912 | 0.920 |
| Plain text | large | 50 | No | - | Relaxed unique | 0.892 | 0.918 | 0.905 |
| Clean PDF | small | 50 | No | PyMuPDF | Strict | 0.708 | 0.849 | 0.772 |
| Clean PDF | small | 50 | No | PyMuPDF | Relaxed occurrences | 0.708 | 0.849 | 0.772 |
| Clean PDF | small | 50 | No | PyMuPDF | Relaxed unique | 0.672 | 0.878 | 0.761 |
| Scanned PDF, no OCR | small | 10 | No | PyMuPDF | Strict | 0.000 | 0.000 | 0.000 |
| Scanned PDF, no OCR | small | 10 | No | PyMuPDF | Relaxed occurrences | 0.000 | 0.000 | 0.000 |
| Scanned PDF, no OCR | small | 10 | No | PyMuPDF | Relaxed unique | 0.000 | 0.000 | 0.000 |
| Matched chat subset | small | 2 | No | - | Strict | 0.647 | 0.805 | 0.717 |
| Matched chat subset | small | 2 | No | - | Relaxed occurrences | 0.647 | 0.805 | 0.717 |
| Matched chat subset | small | 2 | No | - | Relaxed unique | 0.625 | 0.865 | 0.726 |
| Matched chat subset | large | 2 | No | - | Strict | 0.927 | 0.927 | 0.927 |
| Matched chat subset | large | 2 | No | - | Relaxed occurrences | 0.927 | 0.927 | 0.927 |
| Matched chat subset | large | 2 | No | - | Relaxed unique | 0.889 | 0.923 | 0.906 |
| Chat subset | Codex chat + alignment | 2 | No | - | Strict | 1.000 | 1.000 | 1.000 |
| Chat subset | Codex chat + alignment | 2 | No | - | Relaxed occurrences | 1.000 | 1.000 | 1.000 |
| Chat subset | Codex chat + alignment | 2 | No | - | Relaxed unique | 1.000 | 1.000 | 1.000 |

### Difficulties and decisions

PDF extraction can change whitespace and character positions, so strict scores can mix extraction alignment errors with NER errors. In this supplied clean-PDF dataset, all 50 extracted texts have the same length as their plain-text counterparts and become identical after whitespace normalization; the observed entity coordinates remain aligned. Relaxed normalization removes whitespace differences but does not correct spelling, accents or changed entity boundaries. The ten scanned PDFs contain images without a text layer, and direct PyMuPDF extraction returns empty text. They are kept as an explicit no-OCR baseline, with empty predictions and all gold mentions counted as false negatives. OCR would be needed to make these scans usable for NER; no additional extraction library was added.

The windows retain character coordinates in the original input, and matching by id prevents the ten scanned files from being accidentally evaluated against the wrong gold records. Five automated tests cover strict/relaxed scoring, repeated mentions, id alignment, overlapping windows, PDF extraction and chat span alignment. All notebooks were executed successfully.

### Conclusions

Small GLiNER obtained strict micro F1 0.772 and large obtained 0.920 on the same 50 texts; large was higher in this run. The hardest entity type for small was ADDRESS (F1 0.040); PERSON_NAME and DATE were much stronger (F1 0.973 and 0.983). Of 200 gold address mentions, 186 had a same-label overlapping prediction with incorrect boundaries, and two had no same-label overlap. Amount units and organization boundaries also caused errors.

Clean-PDF and plain-text small-model scores were identical. Whitespace changes preserved the annotated coordinates here, so strict and relaxed-occurrence F1 also coincide. Relaxed unique F1 differs because repeated mentions are weighted once per document; it is not guaranteed to exceed occurrence-based F1. This dataset does not demonstrate a strict-score penalty from clean-PDF extraction. In other layouts, shifted offsets can reduce strict scores despite matching entity texts.

Scanned-PDF F1 was zero in all views because extraction produced no text. This result measures a missing extraction stage, not GLiNER's ability to recognize entities from readable text. The chat-plus-alignment experiment matched all 82 gold mentions in its two-document sample, giving F1 1.000. This is a small, template-like sample with exact-string alignment assistance; it cannot establish general superiority over GLiNER. The table includes both GLiNER models evaluated on those same two documents to make the limited comparison explicit.

### Repository and references

Public repository: [AlexGinabreda-Ins/Ner_Project](https://github.com/AlexGinabreda-Ins/Ner_Project).

Implementation references: [GLiNER basic usage](https://github.com/urchade/GLiNER#basic-usage), [large model card](https://huggingface.co/gliner-community/gliner_large-v2.5), and [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html).
