# Chat extraction protocol

Session: current Codex chat, 15 November 2026. No API requests were used for this experiment. The exact underlying model identifier is not exposed in the project.

Input: text fields of test-0001 and test-0002, selected before scoring (one Spanish, one Catalan).

Extraction instruction: Read each administrative document. Identify the unique literal values for PERSON_NAME, ID_CARD_NUMBER, ADDRESS, REGISTRATION_NUMBER, DATE, AMOUNT and ORGANIZATION. Preserve accents, punctuation, currency units and the complete surface form of each mention. Return JSONL objects with id and entities, each entity containing text and label. Do not infer missing values.

The entity lists were selected by the assistant in this chat. The saved local response is chat_response.jsonl. ner.llm aligns each literal value to all exact occurrences in its source text, computes start/end offsets, and then evaluates. It does not read gold entities to construct predictions. This alignment assistance means the strict score is not a test of unaided LLM character counting.

Documents, response and full predictions remain local at the user's request. A fresh clone needs a new chat response to reproduce this experiment. Aggregate scores are published.
