import json
import conf

def format_card_for_bert(card):
    """
    Formats a Magic: The Gathering card dictionary into a string suitable for BERT input.

    Args:
        card (dict): A dictionary containing card details.

    Returns:
        str: A formatted string combining all relevant card fields.
    """
    name = card.get("name", "")
    mana_cost = card.get("mana_cost", "")
    cmc = str(card.get("cmc", ""))
    type_line = card.get("type_line", "")
    oracle_text = card.get("oracle_text", "").replace("\n", " ")
    power = card.get("power", "")
    toughness = card.get("toughness", "")
    colors = ", ".join(card.get("colors", []))

    # Flatten keyword dictionary into readable form
    keywords = card.get("keywords", [])
    keyword_descriptions = []
    for kw in keywords:
        for k, v in kw.items():
            keyword_descriptions.append(f"{k}: {v}")
    keyword_text = " ".join(keyword_descriptions)

    # Combine all fields into a single text
    formatted = (
        f"Name: {name}. "
        f"Mana Cost: {mana_cost}. "
        f"CMC: {cmc}. "
        f"Type: {type_line}. "
        f"Oracle Text: {oracle_text}. "
        f"Power/Toughness: {power}/{toughness}. "
        f"Colors: {colors}. "
        f"Keywords: {keyword_text}."
    )

    return formatted.strip()

# if __name__ == "__main__":
#     # Example card data
#     INPUT_FILE = conf.processed_json
#     with open(INPUT_FILE, "r", encoding="utf-8") as f:
#         cards = json.load(f)
#     example_card = cards[3123]  # Get the first card for demonstration

#     formatted_card = format_card_for_bert(example_card)
#     print(formatted_card)