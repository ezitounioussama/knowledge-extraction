"""The source document, kept in one place so every step reads the same text."""

DOCUMENT = (
    "The Eiffel Tower, located in Paris, France, was constructed between 1887 and 1889 "
    "as the entrance arch to the 1889 World's Fair.\n"
    "It was designed by Gustave Eiffel's engineering company and initially faced criticism "
    "from artists and intellectuals who considered it an eyesore.\n"
    "Today, the Eiffel Tower is one of the most visited monuments in the world, attracting "
    "over 7 million visitors annually.\n"
    "In 2015, special lighting systems were added to enhance its nighttime appearance and "
    "improve energy efficiency."
)

# What a careful human reader would extract, used to score the automated output.
# Two entries are deliberate traps:
#   - the designer is Gustave Eiffel's COMPANY, not the man himself
#   - the visitor figure is "over 7 million", not "7 million"
GOLD_FACTS = {
    "subject": "The Eiffel Tower",
    "location": "Paris, France",
    "construction_start": "1887",
    "construction_end": "1889",
    "purpose": "entrance arch to the 1889 World's Fair",
    "designer": "Gustave Eiffel's engineering company",
    "initial_reception": "criticism from artists and intellectuals who considered it an eyesore",
    "visitors": "over 7 million annually",
    "lighting_year": "2015",
    "lighting_purpose": "enhance nighttime appearance and improve energy efficiency",
}
