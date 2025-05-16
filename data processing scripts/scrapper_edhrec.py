from asyncio import sleep
import json
import os
from pyedhrec import EDHRec
import conf

def get_relevant_synergy_cards(card_name, min_synergy=0.1, min_inclusion=5000, card_types=["highsynergycards", "topcards"]):
    # Ensure tag list is lowercase
    card_types = [t.lower() for t in card_types]

    # Load from file
    file_path = f"edhrec_data/raw/{card_name}_commander_data.json"
    if not os.path.exists(file_path):
        edhrec = EDHRec()
        commander_data = edhrec.get_card_details(card_name)
        os.makedirs("edhrec_data", exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(commander_data, f, indent=4)


    with open(file_path, "r") as f:
        commander_data = json.load(f)

    result_cards = []
    cardlists = commander_data.get("container", {}).get("json_dict", {}).get("cardlists", [])

    for category in cardlists:
        tag = category.get("tag", "").lower()
        if tag not in card_types:
            continue

        for card in category.get("cardviews", []):
            synergy = card.get("synergy", 0)
            inclusion = card.get("inclusion", 0)
            num_decks = card.get("num_decks", 0)

            if synergy >= min_synergy and inclusion >= min_inclusion:
                result_cards.append({
                    "name": card.get("name"),
                    "synergy": round(synergy, 4),
                    "inclusion": inclusion,
                    "num_decks": num_decks,
                    "potential_decks": card.get("potential_decks", 0),
                    "tag": tag,
                })

    return result_cards

def classify_synergy_binary(cards, synergy_positive_threshold=0.3, synergy_negative_threshold=0.1):
    """
    Classifies each card in the list as synergistic (1) or not (0).
    
    Args:
        cards (list): Output from get_relevant_synergy_cards().
        synergy_positive_threshold (float): Minimum synergy to consider a card synergistic.
        synergy_negative_threshold (float): Maximum synergy to consider a card non-synergistic.
        
    Returns:
        List of dicts with binary label added: {"name": ..., "synergy": ..., "label": 0 or 1}
    """
    labeled = []
    for card in cards:
        tag = card.get("tag", "")
        synergy = card.get("synergy", 0)

        if tag == "highsynergycards":
            label = 1
        elif tag == "topcards":
            if synergy >= synergy_positive_threshold:
                label = 1
            elif synergy <= synergy_negative_threshold:
                label = 0
            else:
                continue  # skip ambiguous cases
        else:
            continue  # skip unknown tags

        labeled.append({**card, "label": label})

    return labeled

def get_valid_commanders(embedded_data):
    """
    Returns a list of valid commander card names from a JSON file.
    A card is valid if it is a legendary creature or has commander-enabling mechanics.
    """

    valid_commanders = []
    for card in embedded_data:
        type_line = card.get("type_line", "").lower()
        oracle_text = card.get("oracle_text", "").lower()

        # Basic rule: legendary creature
        if "legendary" in type_line and "creature" in type_line:
            valid_commanders.append(card["name"])
            continue

        # Alternate rule: text mentions it can be your commander
        if "can be your commander" in oracle_text:
            valid_commanders.append(card["name"])
            continue

        # Optional rule: has keywords related to commander mechanics
        commander_keywords = {"partner", "choose a background", "background"}
        if any(kw in oracle_text for kw in commander_keywords):
            valid_commanders.append(card["name"])

    return valid_commanders




if __name__ == "__main__":
    embedded_data = open(conf.embedding_file, "r")
    embedded_data = json.load(embedded_data)

    # Get relevant synergy cards
    cards_names = get_valid_commanders(embedded_data)
    print(f"Found {len(cards_names)} valid commanders.")

    for card_name in cards_names:
        path_label = f"edhrec_data/labeled/{card_name}_labeled.json"
        if os.path.exists(path_label):
            print(f"Card {card_name} already exists. Skipping...")
            continue
        
        sleep(0.2)

        print(f"Processing {card_name}...")
        relevant_cards = get_relevant_synergy_cards(card_name)
        labeled_cards = classify_synergy_binary(relevant_cards)

        # Save to file
        with open(path_label, "w") as f:
            json.dump(labeled_cards, f, indent=4)
