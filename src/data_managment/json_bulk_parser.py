import json
import time
import mtgProject.src.utils.conf as conf


def print_all_layouts(cards):
    """
    Print all unique layouts from the given card data.

    Args:
        cards (list): List of card dictionaries
    """
    layouts = set()
    fieldsnames_in_card_faces = set()
    objects_in_card_faces = set()
    for card in cards:
        if card.get("legalities", {}).get("commander") == "legal":
            layout = card.get("layout")

            if layout:
                card_faces = card.get("card_faces", [])
                if layout not in layouts:
                    layouts.add(layout)
                    print(f"Card: {card.get('name')}, Layout: {layout}, Card Faces: {json.dumps([face["name"] for face in card_faces], indent=2)}")
                    print("--------------------------\n\n")


                #print the card name and layout and card_faces
                
                if card_faces:
                    # Collect all unique field names in card_faces
                    for face in card_faces:
                        for key in face.keys():
                            #if key is a object, expand it
                            if isinstance(face[key], dict):
                                for sub_key in face[key].keys():
                                    fieldsnames_in_card_faces.add(f"{key}.{sub_key}")

                            if key == "object":
                                objects_in_card_faces.add(face[key])
                            fieldsnames_in_card_faces.add(key)

                    
                # else:
                #     print(f"Card: {card.get('name')}, Layout: {layout}, Card Faces: None")

                    

    print("Unique layouts found:")
    for layout in sorted(layouts):
        print(f"- {layout}")
    print("\nFields in card_faces:")
    for field in sorted(fieldsnames_in_card_faces):
        print(f"- {field}")
    print("\nObjects in card_faces:")
    for obj in sorted(objects_in_card_faces):
        print(f"- {obj}")

def handle_normal_cards(card):
    """
    Handle normal cards (non-special layouts) and return a simplified card object.

    Args:
        card (dict): Card dictionary from Scryfall JSON

    Returns:
        dict: Simplified card object
    """
    return {
        "name": card.get("name"),
        "set": card.get("set"),
        "mana_cost": card.get("mana_cost", ""),
        "cmc": card.get("cmc", 0),
        "type_line": card.get("type_line", ""),
        "oracle_text": card.get("oracle_text", ""),
        "power": card.get("power", ""),
        "toughness": card.get("toughness", ""),
        "colors": card.get("colors", []),
        "keywords": card.get("keywords", []),
        "collector_number": card.get("collector_number", ""),
        "color_identity": card.get("color_identity", []),
        "image_uris": card.get("image_uris", {}),
        "layout": card.get("layout", "normal"),
    }

def handle_special_layouts(card, special_layouts_noflip):
    """
    Handle special card layouts (split, transform, flip, modal_dfc, adventure).
    These cards include multiple faces that must be stored and interpreted properly.

    Args:
        card (dict): Scryfall card entry with special layout

    Returns:
        dict: Processed card with `card_faces` field
    """
    layout = card.get("layout", "")
    card_faces = card.get("card_faces", [])
    result_faces = []

    for face in card_faces:
        face_obj = {
            "name": face.get("name", ""),
            "mana_cost": face.get("mana_cost", ""),
            "type_line": face.get("type_line", ""),
            "oracle_text": face.get("oracle_text", ""),
            "power": face.get("power", ""),
            "toughness": face.get("toughness", ""),
            "loyalty": face.get("loyalty", ""),
            "colors": face.get("colors", []),
        }
        if layout not in special_layouts_noflip:
            face_obj["image_uris"] = face.get("image_uris", {})
        result_faces.append(face_obj)

    simplified = {
        "name": card.get("name"),
        "set": card.get("set"),
        "cmc": card.get("cmc", 0),
        "collector_number": card.get("collector_number", ""),
        "color_identity": card.get("color_identity", []),
        "keywords": card.get("keywords", []),
        "layout": layout,
        "card_faces": result_faces,
    }

    if layout in special_layouts_noflip:
        # For modal double-faced cards, we need to include the `image_uris` from the first face
        if card_faces:
            simplified["image_uris"] = card.get("image_uris", {})

    return simplified

def filter_commander_legal_cards_and_process(
    input_file, output_file, set_list, indent=1, output_dir=""
):
    """
    Process a Scryfall JSON file and create a new JSON with commander-legal cards from specified sets.

    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        set_list (list): List of set codes to include (e.g., ['drc', 'cmr'])
    """

    special_layouts = [
        "transform",
        "modal_dfc",
        "split",
        "flip",
        "adventure",
    ]

    special_layouts_noflip = [
        "flip",
        "split",
        "adventure",
    ]
    try:
        # Load the input JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            cards = json.load(f)

        # Filter cards
        filtered_cards = []
        missing_keywords = []
        for card in cards:
            # Skip if not in our set list or not commander legal
            if (
                card.get("set") not in set_list
                or card.get("legalities", {}).get("commander") != "legal"
            ):
                continue

            layout = card.get("layout", "")
            if layout in special_layouts:
                # Handle special layouts
                filtered_card = handle_special_layouts(card, special_layouts_noflip)
            else:
                # Handle normal cards
                filtered_card = handle_normal_cards(card)

            # channge keyword field from list of string to list of object of the type {keyword: description}
            new_keywords = []
            for keyword in filtered_card["keywords"]:
                if keyword in conf.MTG_KEYWORDS:
                    new_keywords.append({keyword: conf.MTG_KEYWORDS[keyword]})
                else:
                    if keyword not in missing_keywords:
                        missing_keywords.append(keyword)

            filtered_card["keywords"] = new_keywords
            filtered_cards.append(filtered_card)

        match indent:
            case 0:
                save_cards(filtered_cards, output_file, output_dir, indent_bol=False)
            case 1:
                save_cards(filtered_cards, output_file, output_dir, indent_bol=True)
            case 2:
                save_cards(filtered_cards, output_file, output_dir, indent_bol=True)
                save_cards(filtered_cards, output_file, output_dir, indent_bol=False)

        # Save missing keywords to a separate file, if present already overwrite the file
        if missing_keywords:
            with open(output_dir + "missing_keywords.json", "w", encoding="utf-8") as f:
                json.dump(missing_keywords, f, indent=4, ensure_ascii=False)
        else:
            print("No missing keywords found.")

        print(f"Successfully processed {len(filtered_cards)} cards to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def save_cards(cards, output_file, output_dir, indent_bol=True):
    """
    Save the filtered cards to a new JSON file.

    Args:
        cards (list): List of filtered card dictionaries
        output_file (str): Path to output JSON file
        output_dir (str): Directory for output files
        indent (bool): Whether to use indentation in the JSON file
    """
    if indent_bol:
        with open(output_dir + "indent/" + output_file, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=4, ensure_ascii=False)
    else:
        with open(output_dir + "no_indent/" + output_file, "w", encoding="utf-8") as f:
            json.dump(cards, f, ensure_ascii=False)

def extract_all_sets_from_file(bulk_file):
    """
    Extract all sets from the bulk file.

    Args:
        bulk_file (str): Path to the bulk file
    """
    try:
        with open(bulk_file, "r", encoding="utf-8") as f:
            cards = json.load(f)

        sets = set()
        for card in cards:
            sets.add(card.get("set"))

        return sets
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []
    
def extract_all_sets(bulk_data):
    sets = set()
    for card in bulk_data:
        sets.add(card.get("set"))

    return list(sets)

    


if __name__ == "__main__":
    date = time.strftime("%Y%m%d%H%M%S")
    input_json = conf.bulk_file
    output_json = "commander_legal_cards" + date + ".json"  # Output file
    output_dir = "datasets/processed/"

    sets_to_include = conf.all_sets

    # with open(input_json, "r", encoding="utf-8") as f:
    #     cards = json.load(f)

    # print_all_layouts(cards)
    

    filter_commander_legal_cards_and_process(
        input_json, output_json, sets_to_include, indent=2, output_dir=output_dir
    )

    # print(extract_all_sets(input_json))
