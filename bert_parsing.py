import json
import conf

def format_card_for_bert(card):
    """
    Formats a Magic: The Gathering card dictionary into a string suitable for BERT input,
    including special layout handling for split, transform, modal_dfc, etc.

    Args:
        card (dict): A dictionary containing card details from Scryfall.

    Returns:
        str: A formatted string combining all relevant card fields.
    """
    layout = card.get("layout", "normal")
    special_layouts = {"split", "transform", "modal_dfc", "flip", "adventure"}

    def flatten_keywords(keywords):
        keyword_descriptions = []
        for kw in keywords:
            for k, v in kw.items():
                keyword_descriptions.append(f"{k}: {v}")
        return " ".join(keyword_descriptions)

    def format_face(face):
        name = face.get("name", "")
        mana_cost = face.get("mana_cost", "")
        type_line = face.get("type_line", "")
        oracle_text = face.get("oracle_text", "").replace("\n", " ")
        power = face.get("power", "")
        toughness = face.get("toughness", "")
        return f"[Face: {name}, Mana Cost: {mana_cost}, Type: {type_line}, Oracle Text: {oracle_text}, Power/Toughness: {power}/{toughness}]"

    if layout in special_layouts and "card_faces" in card:
        face_texts = [format_face(face) for face in card["card_faces"]]
        face_block = " ".join(face_texts)
        cmc = str(card.get("cmc", ""))
        colors = ", ".join(card.get("colors", []))
        color_identity = ", ".join(card.get("color_identity", []))
        keywords = flatten_keywords(card.get("keywords", []))

        formatted = (
            f"Name: {card.get('name', '')}. "
            f"CMC: {cmc}. "
            f"Layout: {layout}. "
            f"Colors: {colors}. "
            f"Color Identity: {color_identity}. "
            f"Keywords: {keywords}. "
            f"Faces: {face_block}."
        )
    else:
        name = card.get("name", "")
        mana_cost = card.get("mana_cost", "")
        cmc = str(card.get("cmc", ""))
        type_line = card.get("type_line", "")
        oracle_text = card.get("oracle_text", "").replace("\n", " ")
        power = card.get("power", "")
        toughness = card.get("toughness", "")
        colors = ", ".join(card.get("colors", []))
        color_identity = ", ".join(card.get("color_identity", []))
        keywords = flatten_keywords(card.get("keywords", []))

        formatted = (
            f"Name: {name}. "
            f"Mana Cost: {mana_cost}. "
            f"CMC: {cmc}. "
            f"Type: {type_line}. "
            f"Oracle Text: {oracle_text}. "
            f"Power/Toughness: {power}/{toughness}. "
            f"Colors: {colors}. "
            f"Color Identity: {color_identity}. "
            f"Keywords: {keywords}."
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