# Knowledge Extraction from a Document

Split a document, extract its facts, summarise it — with every extracted value checked back
against the source. Runs locally on Ollama, no API key.

```bash
ollama serve && ollama pull llama3.2:3b
python3 -m venv .venv
.venv/bin/python -m pip install langchain-core langchain-ollama pydantic spacy
.venv/bin/python -m spacy download en_core_web_sm

.venv/bin/python run.py
```

| File | Contents |
|---|---|
| `document.py` | The source text and the gold facts used to score the output |
| `chunking.py` | Task 1 — semantic split with metadata enrichment |
| `facts.py` | Task 2 — schema, extraction, grounding and precision checks |
| `run.py` | All three tasks, with verification |
| `output/knowledge.json` | Chunks, facts and summary as a dataset |
| `docs/output.txt` | Raw run log |

---

## Task 1 — Splitting strategy: semantic, with metadata

Four sentences, four distinct topics, so sentence boundaries are topic boundaries. Split with
spaCy, then each chunk tagged with its entities, a topic label and a time period.

| # | Topic | Period | Entities |
|---|---|---|---|
| 1 | construction and original purpose | 1887–1889 | Paris, France; between 1887 and 1889; 1889 |
| 2 | designer and initial reception | unspecified | Gustave Eiffel's |
| 3 | present-day visitor numbers | present day | over 7 million; Today; annually |
| 4 | 2015 lighting upgrade | modern (2015) | 2015 |

**Why not fixed-size.** The same text cut at 120 characters:

```
[1] '...as the entrance arch to the 1889 World'
[2] "'s Fair. It was designed by..."
[3] '...attractin'
[4] 'g over 7 million visitors annually...'
```

`1889 World` | `'s Fair` and `attractin` | `g over 7 million` — the cuts land inside the facts the
exercise is trying to extract.

## Task 2 — Key facts

| Category | Fact |
|---|---|
| subject | The Eiffel Tower |
| place | Paris, France |
| date | constructed between 1887 and 1889 |
| purpose | entrance arch to the 1889 World's Fair |
| event | the 1889 World's Fair |
| organisation | **Gustave Eiffel's engineering company** |
| reception | criticism from artists and intellectuals; considered an eyesore |
| number | **over 7 million** visitors annually |
| attribute | one of the most visited monuments in the world |
| date | 2015 — special lighting systems added |
| purpose | enhance nighttime appearance, improve energy efficiency |

Two precision points that are easy to get wrong: the document credits the **company**, not
Gustave Eiffel personally, and the figure is **over** 7 million — a lower bound, not a count.

## Task 3 — Summary

> The Eiffel Tower was constructed as the entrance arch to the 1889 World's Fair and was initially
> considered an eyesore by artists and intellectuals. It is now one of the most visited monuments
> in the world, attracting over 7 million visitors annually. Special lighting systems were added to
> the tower in 2015 to enhance its nighttime appearance and improve energy efficiency.

Checked automatically: every number in the summary appears in the source, no invented figures, and
the `over` qualifier on 7 million was kept.

---

## Two checks worth having, and why

**Grounding.** Every extracted value is searched for in the source, and anything absent is
discarded rather than exported. On the recorded run that removed four fabrications:

```
DISCARDED — not in the document (4):
  date          '1909'              (the year the tower was officially opened)
  other         'Maurice Koechlin'  (engineer responsible for the main support columns)
  other         'Stephen Sauvestre' (engineer responsible for the curved shape)
  organisation  'none specified'
```

Those are real facts about the Eiffel Tower — and none of them is in these four sentences. They
came from the model's background knowledge, which is out of scope for extraction and indetectable
without checking against the text.

They appeared because the schema demands at least eight facts. Measured while building this:

| Setting | Real facts found (of 10) | Fabrications |
|---|---|---|
| `min_length=3` | 8 | 0 |
| categories widened for qualitative facts | 6 | 0 |
| `min_length=8` + "be exhaustive" | 9 | **9** |

Widening the categories made the model *swap* facts rather than add them — gaining the reception
fact cost it Paris, 1887 and 2015. Demanding more facts got the coverage and manufactured the
rest. So the working arrangement is to ask broadly and filter hard: quantity pressure produces
invention, and grounding is what makes asking for coverage safe.

**Precision.** Grounding alone passes `7 million`, because those words are in the text. But the
document says *over* 7 million, and dropping the qualifier turns a lower bound into an exact
figure. That is a different failure from invention, so it gets its own check:

```
[IMPRECISE] number  '7 million'
            ! '7 million' drops the qualifier 'over' — the document says over 7 million
```

## Known limitations

- The model still sometimes emits `Gustave Eiffel` as a person credited with the design. Grounding
  passes it, because the name genuinely appears in the text — the error is in the *relationship*,
  not the span, and a substring check cannot see relationships. The prompt reduces it; it does not
  eliminate it. Verifying attribution properly needs a claim-level check against the sentence.
- Some kept facts are low value (`attribute 'improve'`). Grounded and harmless, but noise.
- spaCy NER has its own quirks on this text: `EVENT` came back as `World's` (truncated),
  `PERSON` as `Gustave Eiffel's` (possessive included, and the company did the designing), and
  `CARDINAL` includes `one` from "one of the most visited". NER is a signal to check the model
  against, not ground truth.

---

Author: **Oussama Ezitouni**
