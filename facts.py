"""Task 2 — extract the key facts, typed and verified.

Two independent extractors run over the same chunks, because they fail in
different ways:

  spaCy NER   deterministic and always grounded — it can only return spans that
              exist in the text. But it labels by surface form, so it tags
              "Gustave Eiffel's" as a PERSON even though the sentence says the
              company designed the tower.

  LLM + Pydantic  understands the sentence, so it can answer "who designed it"
              rather than "which names appear". But it can invent, so anything it
              returns is checked back against the source.

Agreement between the two is the useful signal. Where they disagree, the source
text decides.
"""

import re
from typing import List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

MODEL = "qwen3:8b"

# Labels worth keeping from spaCy. CARDINAL is included for the visitor figure;
# ORDINAL and similar are dropped as noise.
USEFUL_LABELS = ("PERSON", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "QUANTITY",
                 "EVENT", "FAC", "NORP")


class Fact(BaseModel):
    """One extracted fact, with the text it came from."""

    category: str = Field(
        ...,
        description=(
            "date, place, person, organisation, number, event, purpose, "
            "reception, or attribute"
        ),
    )
    value: str = Field(..., min_length=1, max_length=200, description="The fact itself")
    detail: Optional[str] = Field(default=None, max_length=300,
                                 description="What this fact says, in a few words")

    @field_validator("category")
    @classmethod
    def normalise_category(cls, value):
        """Collapse the model's spelling variants into one set of labels.

        Left alone the output contains "Date", "date", "DATE" and "dates" as four
        separate categories, which makes the fact table impossible to group.
        """
        mapping = {
            "date": "date", "dates": "date", "time": "date", "year": "date",
            "place": "place", "places": "place", "location": "place", "gpe": "place",
            "person": "person", "people": "person",
            "organisation": "organisation", "organization": "organisation",
            "org": "organisation", "company": "organisation",
            "number": "number", "numbers": "number", "quantity": "number",
            "statistic": "number", "cardinal": "number",
            "event": "event", "events": "event",
            # Non-entity categories. An earlier version stopped at the entity
            # types above, and two facts a careful reader would extract had
            # nowhere to go: the tower's PURPOSE ("entrance arch to the 1889
            # World's Fair") and its RECEPTION ("criticism from artists and
            # intellectuals"). Neither is a date, place, person or number, so the
            # extraction simply dropped them — a schema limitation, not a model
            # failure.
            "purpose": "purpose", "function": "purpose", "reason": "purpose",
            "reception": "reception", "criticism": "reception",
            "opinion": "reception", "reaction": "reception",
            "attribute": "attribute", "description": "attribute",
            "fact": "attribute", "detail": "attribute", "improvement": "purpose",
        }
        key = str(value).strip().lower()
        return mapping.get(key, "other")


class ExtractedFacts(BaseModel):
    """The full fact set for one document."""

    subject: str = Field(..., description="What the document is about")
    # min_length=8 is doing real work here, not just documenting an expectation.
    #
    # With min_length=3 the model returned six or seven facts whatever the prompt
    # said, and widening the category set made it SWAP facts rather than add them:
    # gaining the reception fact cost it Paris, 1887 and 2015. The binding limit
    # was the length of the list, not the schema. Requiring eight forces coverage,
    # and the repair pass catches a short list and asks for the rest.
    facts: List[Fact] = Field(
        ..., min_length=8, max_length=20,
        description="Every distinct fact in the document, not a selection",
    )


parser = JsonOutputParser(pydantic_object=ExtractedFacts)

# temperature=0: extraction should give the same answer every run.
model = ChatOllama(model=MODEL, reasoning=False, temperature=0.0, num_predict=700)

# Two rules earn their place here. "Quote the document's own words" is what keeps
# the values verifiable. The designer instruction is there because the sentence
# says the COMPANY designed the tower, and the obvious reading — that Gustave
# Eiffel did it personally — is wrong.
facts_prompt = PromptTemplate(
    template="""Extract the key facts from the document below.

For each fact give:
- category: one of date, place, person, organisation, number, event, purpose,
  reception, attribute
- value: the fact, quoted using the document's own words wherever possible
- detail: what it means, in a few words

Rules:
- Extract only what the document states. Add nothing from outside knowledge.
- Be exhaustive. Extract EVERY distinct fact, not a representative sample. A short
  document still yields many facts: each date, each place, the subject, the
  purpose, the reception, each figure, and what each change achieved. Aim for at
  least eight.
- Include facts that are not names or numbers: why something was built (purpose),
  how people reacted to it (reception), and what a change was meant to achieve.
  These matter as much as the dates.
- Quote figures exactly as written, including words like "over" or "about".
- Be precise about who did what. If the document credits a company, do not credit
  a person, and the reverse.
- Return JSON only.

{format_instructions}

Document:
{document}
""",
    input_variables=["document"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

facts_chain = facts_prompt | model | parser


summary_prompt = PromptTemplate.from_template(
    """Summarise the document below in 2 to 3 sentences.

Rules:
- Use only what the document states. Do not add facts, dates or figures.
- Keep numbers exactly as written, including qualifiers such as "over".
- Be precise about attribution: if a company is credited, do not credit a person.
- Return the summary only, with no preamble.

Document:
{document}
"""
)

summary_chain = summary_prompt | model


def ner_facts(chunks):
    """Collect spaCy entities across the chunks, keeping useful labels only.

    Returns {label: [values]}. These are grounded by construction: an entity is a
    span of the source, so it cannot be invented — which is exactly why it is
    worth comparing the model against.
    """
    collected = {}

    for chunk in chunks:
        for label, values in chunk.entities.items():
            if label not in USEFUL_LABELS:
                continue
            for value in values:
                collected.setdefault(label, [])
                if value not in collected[label]:
                    collected[label].append(value)

    return collected


def verify_fact(value: str, source: str) -> bool:
    """Is this value actually supported by the source text?

    Exact substring first, then a word-overlap fallback, because the model may
    reasonably write "1887 to 1889" where the document says "between 1887 and
    1889". Requiring an exact match would reject a correct fact; requiring nothing
    would accept an invented one.

    The bar for the fallback is that every content word of 3+ characters appears
    somewhere in the source.
    """
    lowered_source = source.lower()
    lowered_value = value.lower().strip()

    if lowered_value in lowered_source:
        return True

    words = [w for w in re.split(r"\W+", lowered_value) if len(w) >= 3]
    if not words:
        return False

    return all(word in lowered_source for word in words)


# The repair prompt deliberately contains NO example JSON.
#
# A first version showed the required shape as a template with placeholder values
# ("category": "date|place|person...", "detail": "<a few words>"). The model
# returned that template verbatim as its answer — placeholders and all — and
# validation rejected it for having one fact instead of three. A concrete example
# inside an instruction can be read as the content to produce, so the requirements
# are described in words instead.
REPAIR_PROMPT = PromptTemplate.from_template(
    """This JSON was extracted from a document but failed validation.

Problems found:
{errors}

Fix ONLY those problems. Keep every fact that is already there — do not shorten
the list, do not replace values with descriptions of values, and do not return a
template. If a required top-level field is missing, add it with a real value taken
from the facts already present.

Required: a top-level "subject" naming what the document is about, and a "facts"
list of at least eight entries, each with a category, a value and a detail. If the
list is too short, add the facts that are missing — every date, every place, the
purpose, the reception and every figure in the document.

The JSON to fix:
{bad_json}

Return the corrected JSON only.
"""
)

repair_chain = REPAIR_PROMPT | model | parser


def extract_facts(document: str):
    """Extract, validate, and repair once if validation fails.

    Returns (ExtractedFacts, notes). The repair pass exists because a small model
    reliably drops one required field — on the first run of this project it
    omitted `subject` entirely — and re-asking with the error is cheaper and more
    accurate than discarding the whole extraction.
    """
    import json

    from pydantic import ValidationError

    notes = []
    parsed = facts_chain.invoke({"document": document})

    try:
        return ExtractedFacts(**parsed), notes
    except ValidationError as error:
        problems = [
            f"- {'.'.join(str(p) for p in item['loc']) or '(root)'}: {item['msg']}"
            for item in error.errors()
        ]
        notes.append("first attempt failed validation:")
        notes.extend(problems)

    repaired = repair_chain.invoke(
        {"errors": "\n".join(problems), "bad_json": json.dumps(parsed, ensure_ascii=False)}
    )
    facts = ExtractedFacts(**repaired)
    notes.append("repaired on the second attempt")
    return facts, notes


QUALIFIERS = ("over", "about", "approximately", "more than", "nearly", "around",
              "at least", "up to", "almost")


def check_precision(value: str, source: str):
    """Flag a figure that dropped a qualifier the source attached to it.

    The document says "over 7 million visitors annually". A model that returns
    "7 million" is not hallucinating — the words are in the text — so a grounding
    check passes it. But the meaning changed: an unbounded lower bound became an
    exact count.

    This is a different failure from invention and needs its own check. Substring
    verification cannot catch it, because the dropped word is precisely what is
    missing.

    Returns a warning string, or None if the value is faithful.
    """
    lowered_value = value.lower().strip()
    lowered_source = source.lower()

    # Only figures are at risk here.
    if not re.search(r"\d", lowered_value):
        return None

    for qualifier in QUALIFIERS:
        if qualifier in lowered_value:
            return None      # the qualifier was kept

        # Does the source attach this qualifier to this figure?
        pattern = re.escape(qualifier) + r"\s+" + re.escape(lowered_value)
        if re.search(pattern, lowered_source):
            return (
                f"{value!r} drops the qualifier {qualifier!r} — the document says "
                f"{qualifier} {value}"
            )

    return None
