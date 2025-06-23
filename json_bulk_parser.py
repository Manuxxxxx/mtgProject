import json
import time
import conf




def filter_commander_legal_cards(
    input_file, output_file, set_list, indent=1, output_dir=""
):
    """
    Process a Scryfall JSON file and create a new JSON with commander-legal cards from specified sets.

    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        set_list (list): List of set codes to include (e.g., ['drc', 'cmr'])
    """
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

            # Create new card object with only the fields we want
            filtered_card = {
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
            }

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

def extract_all_sets(bulk_file):
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
    


if __name__ == "__main__":
    date = time.strftime("%Y%m%d%H%M%S")
    input_json = conf.bulk_file
    output_json = "commander_legal_cards" + date + ".json"  # Output file
    output_dir = "datasets/processed/"

    sets_to_include = conf.all_sets

    filter_commander_legal_cards(
        input_json, output_json, sets_to_include, indent=2, output_dir=output_dir
    )

    # print(extract_all_sets(input_json))
