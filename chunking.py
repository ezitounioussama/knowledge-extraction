"""Task 1 — the splitting strategy, and why this one.

Three strategies were candidates. The document is four sentences, each carrying a
distinct topic, which decides it:

  Fixed-size      cuts on a character count, so it lands mid-fact. Shown below
                  for comparison, because the damage is the argument.
  Semantic        one chunk per sentence: construction, design and reception,
                  visitors today, the 2015 lighting. Chosen.
  Metadata-based  not a competing split so much as a layer on top — each chunk is
                  tagged with its entities and time period. Applied to the
                  semantic chunks rather than instead of them.

So the strategy used is semantic splitting with metadata enrichment.
"""

import re
from dataclasses import dataclass, field
from typing import List

import spacy

_NLP = None


def nlp():
    """Load en_core_web_sm once. Only the parser and NER are needed."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    return _NLP


@dataclass
class Chunk:
    """One semantic chunk plus the metadata attached to it."""

    index: int
    text: str
    topic: str = ""
    entities: dict = field(default_factory=dict)
    period: str = ""

    @property
    def chars(self) -> int:
        return len(self.text)


def fixed_size_chunks(text: str, size: int = 120) -> List[str]:
    """Included only to show what it costs on this document."""
    flat = " ".join(text.split())
    return [flat[i : i + size] for i in range(0, len(flat), size)]


def semantic_chunks(text: str) -> List[str]:
    """Split on sentence boundaries using spaCy.

    spaCy rather than a regex on `[.!?]`: this document contains "Gustave
    Eiffel's" and figures like "7 million", and a punctuation regex is the wrong
    tool as soon as abbreviations or decimals appear. Here both approaches happen
    to agree, but the choice matters the moment the text gets messier.
    """
    return [sentence.text.strip() for sentence in nlp()(text).sents if sentence.text.strip()]


# Time period per chunk, decided from the dates the chunk actually contains.
def _period_for(entities) -> str:
    dates = entities.get("DATE", [])
    if not dates:
        return "unspecified"

    years = [int(y) for date in dates for y in re.findall(r"\b(1[89]\d{2}|20\d{2})\b", date)]
    if not years:
        return "present day" if any("today" in d.lower() for d in dates) else "unspecified"

    if min(years) < 1900:
        return f"construction era ({min(years)}-{max(years)})"
    return f"modern ({min(years)})"


def _topic_for(text: str) -> str:
    """A short label per chunk, from the words the chunk uses.

    Keyword rules rather than a model call: the labels are for navigating the
    chunks, and a deterministic label is more useful than one that changes
    between runs.
    """
    lowered = text.lower()

    if "constructed" in lowered or "entrance arch" in lowered:
        return "construction and original purpose"
    if "designed" in lowered or "criticism" in lowered:
        return "designer and initial reception"
    if "visited" in lowered or "visitors" in lowered:
        return "present-day visitor numbers"
    if "lighting" in lowered:
        return "2015 lighting upgrade"
    return "other"


def build_chunks(text: str) -> List[Chunk]:
    """Semantic split, then metadata enrichment on each chunk."""
    chunks = []

    for index, sentence in enumerate(semantic_chunks(text), start=1):
        doc = nlp()(sentence)

        entities = {}
        for entity in doc.ents:
            entities.setdefault(entity.label_, [])
            if entity.text not in entities[entity.label_]:
                entities[entity.label_].append(entity.text)

        chunks.append(
            Chunk(
                index=index,
                text=sentence,
                topic=_topic_for(sentence),
                entities=entities,
                period=_period_for(entities),
            )
        )

    return chunks
