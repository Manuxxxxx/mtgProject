import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import conf

BASE_URL = "https://tagger.scryfall.com"


def get_csrf_token(session: requests.Session, set_code: str, card_number: str) -> tuple[str, str]:
    """Fetch the CSRF token and return it with the card page URL"""
    card_url = f"{BASE_URL}/card/{set_code}/{card_number}"
    response = session.get(card_url)
    soup = BeautifulSoup(response.text, "html.parser")
    meta = soup.find("meta", {"name": "csrf-token"})
    if not meta:
        raise RuntimeError("CSRF token not found")
    return meta["content"], card_url


def build_graphql_payload(set_code: str, card_number: str) -> dict:
    """Prepare the GraphQL query and variables"""
    query = """
    query FetchCard(
      $set: String!
      $number: String!
      $back: Boolean = false
      $moderatorView: Boolean = false
    ) {
      card: cardBySet(set: $set, number: $number, back: $back) {
        name
        oracleId
        taggings(moderatorView: $moderatorView) {
          annotation
          foreignKey
          tag {
            name
            category
            ancestorTags {
              name
            }
          }
        }
      }
    }
    """
    variables = {
        "set": set_code,
        "number": card_number,
        "back": False,
        "moderatorView": False
    }
    return {
        "query": query,
        "variables": variables,
        "operationName": "FetchCard"
    }


def fetch_card_data(session: requests.Session, csrf_token: str, card_url: str, payload: dict) -> dict:
    """Send the GraphQL request and return the parsed JSON"""
    headers = {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrf_token,
        "Referer": card_url,
        "Origin": BASE_URL,
        "User-Agent": "Mozilla/5.0"
    }
    response = session.post(f"{BASE_URL}/graphql", headers=headers, json=payload)
    if not response.ok:
        raise RuntimeError(f"GraphQL request failed: {response.status_code}\n{response.text}")
    return response.json()["data"]["card"]


def separate_tags(taggings: list[dict], ignore_non_functional=True) -> tuple[list[dict], list[dict]]:
    """Split taggings into functional and non-functional by foreignKey"""
    functional, non_functional = [], []
    for tagging in taggings:
        tag = tagging["tag"]
        entry = {
            "name": tag["name"],
            "category": tag["category"],
            "ancestors": [a["name"] for a in tag.get("ancestorTags", [])]
        }
        if tagging["foreignKey"] == "oracleId":
            functional.append(entry)
        elif not ignore_non_functional:
            non_functional.append(entry)
    return functional, non_functional

def print_tag_summary(card_name: str, functional: list[dict], non_functional: list[dict]):
    """Nicely format and print tag data"""
    print(f"\nCard: {card_name}")

    print("\nFunctional Tags:")
    for tag in functional:
        print(f" - {tag['name']} ({tag['category']})")
        if tag["ancestors"]:
            print(f"   Ancestors: {', '.join(tag['ancestors'])}")

    print("\nNon-Functional Tags:")
    for tag in non_functional:
        print(f" - {tag['name']} ({tag['category']})")
        if tag["ancestors"]:
            print(f"   Ancestors: {', '.join(tag['ancestors'])}")

def process_card(set_code: str, card_number: str, ignore_non_functional=True):
    """Main wrapper to process a single card"""
    session = requests.Session()
    csrf_token, card_url = get_csrf_token(session, set_code, card_number)
    payload = build_graphql_payload(set_code, card_number)
    card_data = fetch_card_data(session, csrf_token, card_url, payload)
    functional, non_functional = separate_tags(card_data["taggings"], ignore_non_functional)
    
    return functional, non_functional

def process_cards(sets_to_include, bulk_file, storage_file, save_every=50):
    """
    Process multiple cards from the bulk file and store results in the storage file.
    """
    sets_to_process = get_ids_of_sets_to_process(sets_to_include, bulk_file)
    already_processed = get_id_set_already_processed(storage_file)

    unprocessed_cards = get_unprocessed_cards(sets_to_process, already_processed)

    if not unprocessed_cards:
        print("No unprocessed cards found.")
        return

    session = requests.Session()
    csrf_token = None
    processed_count = 0

    for set_code, cards_info in tqdm(unprocessed_cards.items(), desc="Processing all sets", unit="set"):
        for collector_number in tqdm(cards_info["collector_number"], desc=f"Processing {set_code}", unit="card"):
            if csrf_token is None:
                csrf_token, card_url = get_csrf_token(session, set_code, collector_number)

            payload = build_graphql_payload(set_code, collector_number)
            card_data = fetch_card_data(session, csrf_token, card_url, payload)

            functional_tags, non_functional_tags = separate_tags(card_data["taggings"])

            # Store the processed data
            if set_code not in already_processed:
                already_processed[set_code] = {"collector_number": []}
                
            already_processed[set_code]["collector_number"].append(collector_number)
            already_processed[set_code][collector_number] = {
                "name": card_data["name"],
                "functional_tags": functional_tags
            }

            # Print or save the tags as needed
            # print_tag_summary(card_data["name"], functional_tags, non_functional_tags)

            processed_count += 1
            if processed_count % save_every == 0:
                with open(storage_file, "w") as file:
                    json.dump(already_processed, file)
                print(f"Saved progress after processing {processed_count} cards.")

    # Final save after processing all cards
    with open(storage_file, "w") as file:
        #indent at the end
        json.dump(already_processed, file, indent=4)
    print(f"Processed {processed_count} cards. Data saved to {storage_file}.")

def get_unprocessed_cards(sets_to_process, already_processed):
    unprocessed = {}

    for set_code in sets_to_process:
        if set_code not in already_processed:
            # If the set is not processed at all, add all its cards if there are cards to process
            cards_to_process = sets_to_process[set_code]["collector_number"]
            if len(cards_to_process) > 0:
                unprocessed[set_code] = {"collector_number": cards_to_process}
                
                
        else:
            # Find cards in sets_to_process but not in already_processed
            cards_to_process = sets_to_process[set_code]["collector_number"]
            cards_processed = already_processed[set_code]["collector_number"]
            unprocessed_cards = [
                card for card in cards_to_process if card not in cards_processed
            ]

            if len(unprocessed_cards) > 0:
                # Only add the set if there are unprocessed cards
                unprocessed[set_code] = {"collector_number": unprocessed_cards}

    return unprocessed

def get_ids_of_sets_to_process(sets_to_include, bulk_file):
    with open(bulk_file, "r") as file:
        data = json.load(file)

    set_to_process = {}
    for set_code in sets_to_include:
        card_to_process = {"collector_number": []}
        for card in data:
            if card["set"] == set_code:
                card_to_process["collector_number"].append(card["collector_number"])

        set_to_process[set_code] = card_to_process
    return set_to_process

def get_id_set_already_processed(storage_file):
    # Check if the file exists
    try:
        with open(storage_file, "r") as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {}


# Example usage:
if __name__ == "__main__":
    sets_to_include = conf.all_sets
    bulk_file = "datasets/processed/indent/commander_legal_cards20250625183217.json"
    storage_file = "datasets/scryfallTagger_data/store_scrapped_ancestors.json"

    process_cards(sets_to_include, bulk_file, storage_file, save_every=400)
