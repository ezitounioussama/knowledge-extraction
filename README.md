# Knowledge Extraction from a Document

Split a document, pull out its facts, summarise it — and then check every extracted value back
against the source before exporting it.

That last step is the point. Asking a 3B model for at least eight facts from a four-sentence text
produced nine facts, and nine of them were invented — real facts about the Eiffel Tower that
simply are not in these four sentences. Nothing errors when that happens; the output is
well-formed and plausible. So two checks run before anything is written: **grounding**, which
discards any value that does not appear in the source, and **precision**, which catches values
that appear but changed meaning on the way out (`7 million` when the document says *over* 7
million).

Runs locally on Ollama, no API key.

```bash
ollama serve && ollama pull llama3.2:3b
python3 -m venv .venv
.venv/bin/python -m pip install langchain-core langchain-ollama pydantic spacy
.venv/bin/python -m spacy download en_core_web_sm

.venv/bin/python run.py
```

## Also in this repo

- **[NOTES.md](NOTES.md)** — the three tasks with their output, the measured fabrication
  trade-off, and the known limitations
- `output/knowledge.json` — chunks, facts and summary as a dataset
- [`docs/output.txt`](docs/output.txt) — raw run log

---

Author: **Oussama Ezitouni**
