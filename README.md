# NER Project

Named entity recognition in Spanish and Catalan administrative documents.

## Setup

Python 3.13 and uv are required.

```sh
uv sync
uv run python -m unittest discover -s tests
```

The supplied dataset contains 50 plain-text documents, 50 clean PDFs and 10 scanned PDFs. `validation_small.jsonl` contains the first two documents for the chat experiment. The evaluation helpers were supplied with the assignment.
