import json
import os
from time import sleep
from pyedhrec import EDHRec
import mtgProject.src.utils.conf as conf
import re


def get_commander_data(card_name, debug=False):
    edhrec = EDHRec()
    try:
        commander_data = edhrec.get_commander_data(card_name)
        return commander_data
    except Exception as e:
        if debug:
            print(f"Error fetching data for {card_name}: {e}")
        return None


def create_synergy_between_cards_in_deck(edhrec, commander_name, debug=False):
    synergy_data = []
    cardlist = edhrec.get_commanders_average_deck(commander_name)["decklist"]
    cardlist = [re.sub(r"^\d+\s+", "", s) for s in cardlist]
    if debug:
        print(f"shape of cardlists: {len(cardlist)}")
        print(cardlist)

    for i in range(len(cardlist)):
        for j in range(len(cardlist)):
            if i == j:
                continue
            card1 = cardlist[i]
            card2 = cardlist[j]
            if debug:
                print(f"Processing synergy between {card1} and {card2}...")

            new_synergy = {
                "card1": {"name": card1},
                "card2": {"name": card2},
            }

            synergy_data.append(new_synergy)

    return synergy_data


def create_synergy_from_decks(
    commanders_to_process,
    synergy_data,
    synergy_already_created,
    synergy_already_created_file,
    debug=False,
):
    edhrec = EDHRec()
    counter = 0
    for commander in commanders_to_process:
        if debug:
            print(f"Processing commander: {commander}")
        new_synergies = create_synergy_between_cards_in_deck(
            edhrec, commander, debug=debug
        )
        for synergy in new_synergies:
            card1 = synergy["card1"]["name"]
            card2 = synergy["card2"]["name"]
            key = f"{card1}_{card2}"
            if key not in synergy_already_created and key not in synergy_data:
                synergy_already_created[key] = synergy
                counter += 1

                if counter % 400 == 0:
                    if debug:
                        print(f"Processed {counter} synergies so far...")
                    with open(synergy_already_created_file, "w") as f:
                        json.dump(list(synergy_already_created.values()), f, indent=2)

    if debug:
        print(f"Total new synergies created: {counter}")
    with open(synergy_already_created_file, "w") as f:
        json.dump(list(synergy_already_created.values()), f, indent=2)


def deck_synergy(
    synergy_data_file, deck_synergy_data_file, bulk_data_file, debug=False
):
    if not os.path.exists(bulk_data_file):
        print(f"Bulk data file {bulk_data_file} does not exist.")
        return
    with open(bulk_data_file, "r") as f:
        bulk_data = json.load(f)

    if not os.path.exists(synergy_data_file):
        print(f"Synergy data file {synergy_data_file} does not exist.")
        synergy_data = []
    else:
        with open(synergy_data_file, "r") as f:
            synergy_data = json.load(f)

    if debug:
        print(f"Loaded {len(synergy_data)} synergy pairs from {synergy_data_file}")

    if not os.path.exists(deck_synergy_data_file):
        print(
            f"Deck synergy data file {deck_synergy_data_file} does not exist. Creating new file."
        )
        deck_synergy_data = []
    else:
        with open(deck_synergy_data_file, "r") as f:
            deck_synergy_data = json.load(f)

    synergy_data_hash = {
        f"{synergy['card1']['name']}_{synergy['card2']['name']}": synergy
        for synergy in synergy_data
    }

    deck_synergy_data_hash = {
        f"{synergy['card1']['name']}_{synergy['card2']['name']}": synergy
        for synergy in deck_synergy_data
    }

    all_commanders = get_valid_commanders(bulk_data)

    create_synergy_from_decks(
        all_commanders,
        synergy_data_hash,
        deck_synergy_data_hash,
        debug=debug,
    )


def extract_and_label_cards(
    commander_data,
    synergy_positive_threshold=0.4,
    synergy_negative_threshold=0.2,
    min_inclusion=500,
    debug=False,
):
    labeled = []
    cardlists = (
        commander_data.get("container", {}).get("json_dict", {}).get("cardlists", [])
    )

    if debug:
        print(f"Processing commander: {commander_data.get('name')}")
    if not cardlists:
        if debug:
            print(f"No cardlists found for commander: {commander_data.get('name')}")
        return labeled

    for category in cardlists:
        tag = category.get("tag", "").lower()
        if tag not in ["highsynergycards", "topcards"]:
            continue

        for card in category.get("cardviews", []):
            synergy = card.get("synergy", 0)
            inclusion = card.get("inclusion", 0)
            num_decks = card.get("num_decks", 0)
            if debug:
                print(
                    f"Processing card: {card.get('name')} with synergy: {synergy}, inclusion: {inclusion}, num_decks: {num_decks}"
                )

            if inclusion < min_inclusion:
                continue

            if tag == "highsynergycards":
                label = 1
            elif tag == "topcards":
                if synergy >= synergy_positive_threshold:
                    label = 1
                elif synergy <= synergy_negative_threshold:
                    label = 0
                else:
                    continue
            else:
                continue

            labeled.append(
                {
                    "card1": {"name": commander_data.get("name")},
                    "card2": {
                        "name": card.get("name"),
                        "synergy": round(synergy, 4),
                        "inclusion": inclusion,
                        "num_decks": num_decks,
                        "potential_decks": card.get("potential_decks", 0),
                        "tag": tag,
                    },
                    "synergy": label,
                }
            )

    return labeled


def get_valid_commanders(embedded_data):
    valid_commanders = []
    for card in embedded_data:
        type_line = card.get("type_line", "").lower()
        oracle_text = card.get("oracle_text", "").lower()

        if "legendary" in type_line and "creature" in type_line:
            valid_commanders.append(card["name"])
            continue
        if "can be your commander" in oracle_text:
            valid_commanders.append(card["name"])
            continue
        commander_keywords = {"partner", "choose a background", "background"}
        if any(kw in oracle_text for kw in commander_keywords):
            valid_commanders.append(card["name"])

    return valid_commanders


def get_synergy_commander():
    DEBUG = False

    with open(conf.embedding_file, "r") as f:
        embedded_data = json.load(f)

    commanders = get_valid_commanders(embedded_data)
    print(f"Found {len(commanders)} valid commanders.")
    path_label = "edhrec_data/labeled/combined_commander_synergy_data.json"

    all_data = []
    if os.path.exists(path_label):
        with open(path_label, "r") as f:
            all_data = json.load(f)
        existing_commanders = {data["commander"] for data in all_data}
        commanders = [cmd for cmd in commanders if cmd not in existing_commanders]
        print(f"Found {len(commanders)} new commanders to process.")
    else:
        print("No existing data found. Processing all commanders.")

    for card_name in commanders:
        raw_path = f"edhrec_data/raw/{card_name}_commander_data.json"
        commander_data = None
        if DEBUG:
            print(f"Processing {card_name}...")
        if os.path.exists(raw_path):
            commander_data = json.load(open(raw_path, "r"))
            if DEBUG:
                print(f"Loaded cached data for {card_name}.")
        else:
            sleep(0.2)
            commander_data = get_commander_data(card_name)
            if commander_data is None:
                if DEBUG:
                    print(f"Failed to fetch data for {card_name}. Skipping...")
                sleep(0.5)
                continue
            else:
                # Save raw data to avoid re-fetching
                os.makedirs("edhrec_data/raw", exist_ok=True)
                with open(raw_path, "w") as f:
                    json.dump(commander_data, f, indent=2)

        labeled_cards = extract_and_label_cards(commander_data)

        if labeled_cards.__len__() == 0:
            if DEBUG:
                print(f"No labeled cards found for {card_name}. Skipping...")
            continue
        else:
            if DEBUG:
                print(f"Found {len(labeled_cards)} labeled cards for {card_name}.")
            all_data.append(
                {
                    "commander": card_name,
                    "raw_data": commander_data,
                    "labels": labeled_cards,
                }
            )

    # Save combined data
    os.makedirs("edhrec_data", exist_ok=True)
    with open(path_label, "w") as f:
        json.dump(all_data, f, indent=2)

    print(
        f"\n✅ Saved data for {len(all_data)} commanders to edhrec_data/combined_commander_synergy_data.json"
    )


if __name__ == "__main__":
    # get_synergy_commander()
    # edhrec = EDHRec()
    # create_synergy_between_cards_in_deck(edhrec, "Atraxa, Praetors' Voice", debug=True)

    deck_synergy(
        synergy_data_file="edhrec_data/synergy_data.json",
        deck_synergy_data_file="edhrec_data/deck_synergy_data.json",
        bulk_data_file=conf.embedding_file,
        debug=True,
    )
