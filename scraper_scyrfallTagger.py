import time
import requests
import json
import conf


def get_card_info(set_code, card_number, X_CSRF_Token, cookie, debug=False):
    url = "https://tagger.scryfall.com/graphql"

    # Headers from the curl command (including dynamic Referer)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0",
        "Accept": "*/*",
        "Accept-Language": "it-IT,it;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Referer": f"https://tagger.scryfall.com/card/{set_code}/{card_number}",
        "X-CSRF-Token": X_CSRF_Token,
        "Content-Type": "application/json",
        "Origin": "https://tagger.scryfall.com",
        "Connection": "keep-alive",
        "Cookie": cookie,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Priority": "u=4",
        "TE": "trailers",
    }

    # GraphQL query and variables
    graphql_query = """
    query FetchCard($set:String! $number:String! $back:Boolean=false $moderatorView:Boolean=false){
        card:cardBySet(set:$set number:$number back:$back){
            ...CardAttrs
            backside
            layout
            scryfallUrl
            sideNames
            twoSided
            rotatedLayout
            taggings(moderatorView:$moderatorView){
                ...TaggingAttrs
                tag{
                    ...TagAttrs
                    ancestorTags{
                        ...TagAttrs
                    }
                }
            }
            relationships(moderatorView:$moderatorView){
                ...RelationshipAttrs
            }
        }
    }
    fragment CardAttrs on Card{
        artImageUrl
        backside
        cardImageUrl
        collectorNumber
        id
        illustrationId
        name
        oracleId
        printingId
        set
    }
    fragment RelationshipAttrs on Relationship{
        classifier
        classifierInverse
        annotation
        subjectId
        subjectName
        createdAt
        creatorId
        foreignKey
        id
        name
        pendingRevisions
        relatedId
        relatedName
        status
        type
    }
    fragment TagAttrs on Tag{
        category
        createdAt
        creatorId
        id
        name
        namespace
        pendingRevisions
        slug
        status
        type
    }
    fragment TaggingAttrs on Tagging{
        annotation
        subjectId
        createdAt
        creatorId
        foreignKey
        id
        pendingRevisions
        type
        status
        weight
    }
    """

    payload = {
        "query": graphql_query.strip(),
        "operationName": "FetchCard",
        "variables": {
            "set": set_code,
            "number": str(card_number),
            "back": False,
            "moderatorView": False,
        },
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        card_tagggins = data["data"]["card"]["taggings"]
        processed_tags = []

        # 	"1": {
        # 		"annotation": null,
        # 		"subjectId": "88600bd4-4dcc-4788-bba4-78b6cf5ad8f8",
        # 		"createdAt": "2025-01-24T06:04:52Z",
        # 		"creatorId": "732149b0-331a-4527-9335-466be0314fdd",
        # 		"foreignKey": "oracleId",
        # 		"id": "ce7e1543-24dc-429b-9147-e2c021f37aca",
        # 		"pendingRevisions": false,
        # 		"type": "TAGGING",
        # 		"status": "GOOD_STANDING",
        # 		"weight": "MEDIAN",
        # 		"tag": {
        # 			"category": false,
        # 			"createdAt": "2025-05-14T11:47:00Z",
        # 			"creatorId": "1415a95b-0916-4a72-9643-142dbb5374c8",
        # 			"id": "eeee4e98-b0c7-40ed-8d6d-5ebb2958ac15",
        # 			"name": "energy generator",
        # 			"namespace": "card",
        # 			"pendingRevisions": false,
        # 			"slug": "energy-generator",
        # 			"status": "GOOD_STANDING",
        # 			"type": "ORACLE_CARD_TAG",
        # 			"ancestorTags": []
        # 		}
        # 	}
        # }

        for tag_superObj in card_tagggins:
            tag_obj = tag_superObj["tag"]
            if tag_obj["type"] == "ORACLE_CARD_TAG":
                new_tag = {
                    "collector_number": card_number,
                    "set": set_code,
                    "tag": tag_obj["name"],
                    "tag_category": tag_obj["category"],
                    "tag_slug": tag_obj["slug"],
                    "tag_status": tag_obj["status"],
                    "tag_weight": tag_superObj["weight"],
                }

                processed_tags.append(new_tag)

        if debug:
            print("card:", data["data"]["card"]["name"])
            print(json.dumps(processed_tags, indent=2))

        return processed_tags
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


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


def get_unprocessed_cards(sets_to_process, already_processed):
    unprocessed = {}

    for set_code in sets_to_process:
        if set_code not in already_processed:
            # If the set is not processed at all, add all its cards
            unprocessed[set_code] = {
                "collector_number": sets_to_process[set_code]["collector_number"].copy()
            }
        else:
            # Find cards in sets_to_process but not in already_processed
            cards_to_process = sets_to_process[set_code]["collector_number"]
            cards_processed = already_processed[set_code]["collector_number"]
            unprocessed_cards = [
                card for card in cards_to_process if card not in cards_processed
            ]

            if unprocessed_cards:
                unprocessed[set_code] = {"collector_number": unprocessed_cards}

    return unprocessed


def get_remaining_cards(cards_to_process, X_CSRF_Token, cookie, storage_file):
    # Load existing progress
    processed_obj = get_id_set_already_processed(storage_file)

    # Initialize counters
    save_counter = 0
    batch_size = 30  # Save after every 30 cards

    for set_code in cards_to_process:
        print(f"Processing set {set_code}")
        cards_processed = []
        cards_not_processed = []

        # Sort the cards to process numerically
        cards_to_process[set_code]["collector_number"].sort(
            key=lambda x: int(x) if x.isdigit() else float("inf")
        )

        for card_number in cards_to_process[set_code]["collector_number"]:
            print(f"Processing card {card_number} from set {set_code}")

            # Get card info
            card_info = get_card_info(
                set_code,
                card_number,
                X_CSRF_Token=X_CSRF_Token,
                cookie=cookie,
                debug=False,
            )

            if card_info:
                # Initialize set structure if not exists
                if set_code not in processed_obj:
                    processed_obj[set_code] = {"collector_number": []}

                # Store the processed card
                processed_obj[set_code]["collector_number"].append(card_number)
                processed_obj[set_code][card_number] = card_info
                cards_processed.append(card_number)

                # Increment and check save counter
                save_counter += 1
                if save_counter >= batch_size:
                    with open(storage_file, "w") as file:
                        json.dump(processed_obj, file, indent=2)
                    print(f"Saved progress after {save_counter} cards")
                    save_counter = 0
            else:
                cards_not_processed.append(card_number)

            time.sleep(0.2)

        # Save any remaining cards that didn't complete a full batch
        if save_counter > 0:
            with open(storage_file, "w") as file:
                json.dump(processed_obj, file, indent=2)
            print(f"Saved final progress for set {set_code} ({save_counter} cards)")
            save_counter = 0

        print(f"Processed cards: {cards_processed}")
        print(f"Not processed cards: {cards_not_processed}")
        print("------------------------------------\n")


# Example usage
if __name__ == "__main__":

    storage_file = conf.scrapped_store
    bulk_file = conf.bulk_file
    output_scrapped_dir = conf.output_scrapped_dir
    sets_to_include = conf.sets_to_include
    X_CSRF_Token = conf.X_CSRF_Token
    cookie = conf.cookie

    get_remaining_cards(
        get_unprocessed_cards(
            get_ids_of_sets_to_process(sets_to_include, bulk_file),
            get_id_set_already_processed(storage_file),
        ),
        X_CSRF_Token,
        cookie,
        storage_file,
    )
