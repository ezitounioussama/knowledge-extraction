"""
Knowledge extraction from a document: split, extract facts, summarise.

    python3 run.py

Three tasks from the brief, in order, each with its output checked against the
source rather than taken on trust.
"""

import json
import sys
from pathlib import Path

from chunking import build_chunks, fixed_size_chunks
from document import DOCUMENT, GOLD_FACTS
from facts import (
    MODEL,
    check_precision,
    extract_facts,
    ner_facts,
    summary_chain,
    verify_fact,
)

LINE = "=" * 82
THIN = "-" * 82
OUT = Path("output")


def header(title):
    print(f"\n{LINE}\n{title}\n{LINE}")


def section(title):
    print(f"\n{title}\n{THIN}")


def task1_split():
    header("TASK 1 — SPLITTING STRATEGY")

    print(f"""
  Strategy chosen: SEMANTIC (one chunk per sentence), with METADATA enrichment.

  The document is four sentences and each carries a distinct topic, so sentence
  boundaries are also topic boundaries. Metadata is layered on top rather than
  used as an alternative split: every chunk is tagged with its entities, a topic
  label and a time period, which is what makes the chunks addressable later.""")

    section("Why not fixed-size — the same document cut at 120 characters")
    for index, chunk in enumerate(fixed_size_chunks(DOCUMENT, 120), start=1):
        print(f"  [{index}] {chunk!r}")

    print("""
  Chunk 1 ends "the 1889 World" and chunk 2 opens "'s Fair." — the event name is
  cut in half, so neither chunk contains the fact. Chunks 3 and 4 split
  "attractin" / "g over 7 million". Cheap to compute, and it destroys exactly the
  facts this exercise is trying to extract.""")

    chunks = build_chunks(DOCUMENT)

    section(f"Semantic chunks with metadata — {len(chunks)} chunks")
    for chunk in chunks:
        print(f"\n  [{chunk.index}] {chunk.chars} chars")
        print(f"      text     : {chunk.text}")
        print(f"      topic    : {chunk.topic}")
        print(f"      period   : {chunk.period}")
        for label, values in chunk.entities.items():
            print(f"      {label:9}: {values}")

    return chunks


def task2_facts(chunks):
    header("TASK 2 — KEY FACTS")

    print("""
  Two extractors over the same text, because they fail differently:
    spaCy NER  can only return spans that exist, so it never invents — but it
               labels by surface form, not by meaning.
    LLM        understands the sentence, so it can answer "who designed it" — but
               it can invent, so every value is checked back against the source.""")

    section("Extractor A — spaCy NER (grounded by construction)")
    entities = ner_facts(chunks)
    for label, values in entities.items():
        print(f"  {label:10} {values}")

    print("""
  Three things to notice, none of them fatal but all worth knowing:
    - EVENT came back as "World's", truncated: it missed "World's Fair".
    - PERSON is "Gustave Eiffel's" — but the sentence says his COMPANY designed
      the tower. NER tags the name it sees; it does not read the relationship.
    - CARDINAL includes "one", picked up from "one of the most visited".
  NER is a signal to check against, not the answer.""")

    section("Extractor B — LLM with a Pydantic schema")
    facts, notes = extract_facts(DOCUMENT)
    for note in notes:
        print(f"  ! {note}")

    print(f"\n  subject: {facts.subject}\n")

    # Grounding is used as a FILTER here, not only as a label.
    #
    # This is the resolution of a three-way trade-off measured while building the
    # exercise. Asking for at least 3 facts got 8 of 10 real ones and no
    # inventions. Widening the categories gained the reception fact but cost
    # Paris, 1887 and 2015 — the model swapped facts rather than adding them.
    # Demanding at least 8 and "be exhaustive" reached 9 of 10 real facts and
    # manufactured NINE more: 1,100 workers killed, 6,000 stairs, 58 elevators,
    # none of which appear in the document.
    #
    # So the working design is to ask broadly and verify hard: request exhaustive
    # coverage, then discard anything the source does not support. Quantity
    # pressure produces invention, and a filter is what makes that safe.
    kept, rejected = [], []

    for fact in facts.facts:
        grounded = verify_fact(fact.value, DOCUMENT)
        precision = check_precision(fact.value, DOCUMENT)

        record = {
            "category": fact.category,
            "value": fact.value,
            "detail": fact.detail,
            "precision_warning": precision,
        }

        if not grounded:
            rejected.append(record)
        else:
            kept.append(record)

    print(f"\n  KEPT — supported by the source ({len(kept)}):")
    for record in kept:
        mark = "IMPRECISE" if record["precision_warning"] else "ok"
        print(f"    [{mark:9}] {record['category']:13} {record['value']!r}")
        if record["detail"]:
            print(f"                 -> {record['detail']}")
        if record["precision_warning"]:
            print(f"                 ! {record['precision_warning']}")

    if rejected:
        print(f"\n  DISCARDED — not in the document ({len(rejected)}):")
        for record in rejected:
            print(f"    [invented ] {record['category']:13} {record['value']!r}"
                  f"   ({record['detail']})")
        print(f"""
  Every discarded item is a fabrication produced under quantity pressure. None of
  these values appears anywhere in the four sentences — they come from the model's
  background knowledge of the Eiffel Tower, not from the document. Fluent,
  specific, plausible and out of scope, which is exactly why the filter exists.""")

    rows = kept

    section("Checked against what a careful reader would extract")
    print(f"  {'field':22} {'expected':46} found")
    for field, expected in GOLD_FACTS.items():
        joined = " | ".join(r["value"].lower() for r in rows) + " " + facts.subject.lower()
        # A gold fact counts as found if its distinctive words are present.
        key_words = [w for w in expected.lower().replace(",", " ").split() if len(w) > 3]
        hits = sum(1 for w in key_words if w in joined)
        found = "yes" if key_words and hits >= max(1, len(key_words) // 2) else "no"
        print(f"  {field:22} {expected[:44]:46} {found}")

    return facts, rows, entities


def task3_summary():
    header("TASK 3 — SUMMARY")

    summary = summary_chain.invoke({"document": DOCUMENT})
    text = getattr(summary, "content", str(summary)).strip()

    section("Generated summary")
    for line in text.splitlines():
        if line.strip():
            print(f"  {line.strip()}")

    section("Checked against the source")

    # Every number in the summary must appear in the document. A summary is the
    # easiest place for a figure to drift, and the drift is invisible without this.
    import re

    numbers_in_summary = set(re.findall(r"\b\d[\d,\.]*\b", text))
    numbers_in_source = set(re.findall(r"\b\d[\d,\.]*\b", DOCUMENT))
    invented = numbers_in_summary - numbers_in_source

    print(f"  numbers in the summary : {sorted(numbers_in_summary) or 'none'}")
    print(f"  numbers in the source  : {sorted(numbers_in_source)}")
    print(f"  invented numbers       : {sorted(invented) or 'none'}")

    precision = check_precision("7 million", text) if "7 million" in text else None
    if "7 million" in text:
        kept = any(q in text.lower() for q in ("over", "more than"))
        print(f"  kept the 'over' qualifier on 7 million : {kept}")

    # The attribution trap: crediting the man rather than the company.
    lowered = text.lower()
    credits_company = "company" in lowered
    credits_person_only = ("gustave eiffel" in lowered and not credits_company)
    print(f"  credits the company (correct)          : {credits_company}")
    print(f"  credits Gustave Eiffel personally only : {credits_person_only}"
          f"{'   <- attribution error' if credits_person_only else ''}")

    return text


def main():
    header("SETUP")
    print(f"  Model    : {MODEL} via ChatOllama (local, no API key)")
    print(f"  Splitter : spaCy en_core_web_sm sentence boundaries")
    print(f"  Document : {len(DOCUMENT)} characters, 4 sentences")
    print(f"\n  {DOCUMENT.replace(chr(10), chr(10) + '  ')}")

    chunks = task1_split()
    facts, rows, entities = task2_facts(chunks)
    summary = task3_summary()

    # Export, so the result is a dataset rather than terminal output.
    OUT.mkdir(exist_ok=True)
    payload = {
        "document": DOCUMENT,
        "strategy": "semantic (sentence) split with metadata enrichment",
        "chunks": [
            {"index": c.index, "text": c.text, "topic": c.topic,
             "period": c.period, "entities": c.entities}
            for c in chunks
        ],
        "subject": facts.subject,
        "facts": rows,
        "ner_entities": entities,
        "summary": summary,
    }
    (OUT / "knowledge.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    header("EXPORT")
    print(f"  output/knowledge.json   "
          f"{len(chunks)} chunks, {len(rows)} verified facts, 1 summary")

    print(f"\n{LINE}\nDone.\n{LINE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"\nError: {type(error).__name__}: {error}")
        print("Is Ollama running?  ollama serve")
        sys.exit(1)
