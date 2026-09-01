# Knowledge Extraction from a Document

Split a document, pull out its facts, summarise it — and then check every extracted value back
against the source before exporting it.

That last step is the point. Asking `llama3.2:3b` for at least eight facts from a four-sentence
text produced nine facts, nine of them invented — real facts about the Eiffel Tower that simply
are not in these four sentences. Nothing errors when that happens; the output is well-formed and
plausible. So two checks run before anything is written: **grounding**, which discards any value
that does not appear in the source, and **precision**, which catches values that appear but
changed meaning on the way out (`7 million` when the document says *over* 7 million).

On `qwen3:8b` — the model it now runs — nothing gets discarded: 8 facts kept, 0 fabrications, all
10 gold facts found, and the summary keeps the *over* qualifier. The checks stay exactly where
they are. A pipeline that only works because the model happens to be good enough is not a
pipeline that works, and the same code invents four facts the moment you point it at a smaller
model.

Runs locally on Ollama, no API key.

```bash
ollama serve && ollama pull qwen3:8b
uv venv
uv pip install langchain-core langchain-ollama pydantic spacy
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
