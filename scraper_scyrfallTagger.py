import time
import requests
import json
import conf
from tqdm import tqdm


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
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "DNT": "1",
        "Sec-GPC": "1",
        "Content-Length": "1773",  # Adjusted for the actual payload length

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
        print("Fetching card info for set:", set_code, "and card number:", card_number)
        response = requests.post(url, headers=headers, json=payload)
        if(debug):
            print("Request URL:", response.url)
            print("Request Headers:", response.request.headers)
            print("Request Body:", json.dumps(payload, indent=2))
            print("Response Status Code:", response.status_code)
            print("Response Headers:", response.headers)
            print("Response Body:", response.text)
            print(type(response.request.headers))

            return get_card_info(set_code,card_number,X_CSRF_Token,str(response.request.headers["set-cookie"]),debug=False) 
        response.raise_for_status()  # Check for HTTP errors
        if response.text.strip() == "":
            print("Empty response received.")
            return None
        
        data = response.json()
        card_tagggins = data["data"]["card"]["taggings"]
        processed_tags = []

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
from tqdm import tqdm

def get_remaining_cards(cards_to_process, X_CSRF_Token, cookie, storage_file):
    # Load existing progress
    processed_obj = get_id_set_already_processed(storage_file)

    # Initialize counters
    save_counter = 0
    batch_size = 30  # Save after every 30 cards

    for set_code in tqdm(cards_to_process, desc="Processing sets", unit="set"):
        # print(f"Processing set {set_code}")
        cards_processed = []
        cards_not_processed = []

        # Sort the cards to process numerically
        cards_to_process[set_code]["collector_number"].sort(
            key=lambda x: int(x) if x.isdigit() else float("inf")
        )

        for card_number in tqdm(cards_to_process[set_code]["collector_number"], desc=f"Processing cards in {set_code}", unit="card"):
            # print(f"Processing card {card_number} from set {set_code}")

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
                    # print(f"Saved progress after {save_counter} cards")
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

import re
import urllib.parse

def extract_headers_from_curl(curl_text):
    # Extract CSRF token
    csrf_match = re.search(r"-H\s+'X-CSRF-Token:\s*(.+?)'", curl_text)
    csrf_token = csrf_match.group(1) if csrf_match else None

    # Extract raw cookie string
    cookie_match = re.search(r"-H\s+'Cookie:\s*(.+?)'", curl_text)
    raw_cookie = cookie_match.group(1) if cookie_match else ""

    # Unescape URL-encoded characters in cookie string
    decoded_cookie = urllib.parse.unquote(raw_cookie)

    # Split cookies into dictionary
    cookie_dict = dict()
    for pair in decoded_cookie.split("; "):
        if "=" in pair:
            key, value = pair.split("=", 1)
            cookie_dict[key] = value

    # Add CSRF token to the dictionary

    return cookie_dict, csrf_token


# Example usage
if __name__ == "__main__":

    storage_file = conf.scrapped_store
    bulk_file = conf.bulk_file
    output_scrapped_dir = conf.output_scrapped_dir
    sets_to_include = conf.all_sets

    cookie_dict, csrf_token = extract_headers_from_curl('''
        curl 'https://tagger.scryfall.com/graphql' \
  --compressed \
  -X POST \
  -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0' \
  -H 'Accept: */*' \
  -H 'Accept-Language: it-IT,it;q=0.8,en-US;q=0.5,en;q=0.3' \
  -H 'Accept-Encoding: gzip, deflate, br, zstd' \
  -H 'Referer: https://tagger.scryfall.com/card/mh2/3' \
  -H 'X-CSRF-Token: gFb7eW83iD0BZv-DgCD0l4xGQrRkK6ytonVhkT3xsKJbmmVcmgzToKHQMnjH9q-HowPQyjWqi-xk1AUPRvU6cw' \
  -H 'Content-Type: application/json' \
  -H 'Origin: https://tagger.scryfall.com' \
  -H 'DNT: 1' \
  -H 'Sec-GPC: 1' \
  -H 'Connection: keep-alive' \
  -H 'Cookie: _scryfall_tagger_session=eGNmDzbg3sza0R2se22THwxvmOnlojlIe6gW5tPQSGAcKr7ON8e3jGX%2BlXAnQLMPT39SVkS21RFcKOiP4uskJ6ZB461ms0J%2BGUjRcKKFv%2Bf348t6c%2BUDd4PAReDmWQzhrDwD3E9MuzZ9n7NO4xs1TpNN9YBxVFJZvfmTU1oYeDbWzY3Yu0zDHV7eDN%2BAxRtLdk36H2n87v2ue0fsZtk4Ebi1ECG1DyiiNxYHBCCGd05qO55KSPZks5RoPaYzeUf5ZhCsb%2F0TGzXe4HwjHSoIhWvLcu7z6ckoAjQB4CBMVcA%3D--sRImlw7jomQdN%2Fwx--XabMRPIwchddNWHAotYkrQ%3D%3D; _ga_XMVWH04BTD=GS2.1.s1750597699$o1$g0$t1750598005$j60$l0$h0; _ga=GA1.1.808851925.1750597700' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Site: same-origin' \
  -H 'Priority: u=4' \
  -H 'Pragma: no-cache' \
  -H 'Cache-Control: no-cache' \
  -H 'TE: trailers' \
  --data-raw '{"query":"\n        query FetchCard(\n          $set: String!\n          $number: String!\n          $back: Boolean = false\n          $moderatorView: Boolean = false\n        ) {\n          card: cardBySet(set: $set, number: $number, back: $back) {\n            ...CardAttrs\n            backside\n            layout\n            scryfallUrl\n            sideNames\n            twoSided\n            rotatedLayout\n            taggings(moderatorView: $moderatorView) {\n              ...TaggingAttrs\n              tag {\n                ...TagAttrs\n                ancestorTags {\n                  ...TagAttrs\n                }\n              }\n            }\n            relationships(moderatorView: $moderatorView) {\n              ...RelationshipAttrs\n            }\n          }\n        }\n        \n  fragment CardAttrs on Card {\n    artImageUrl\n    backside\n    cardImageUrl\n    collectorNumber\n    id\n    illustrationId\n    name\n    oracleId\n    printingId\n    set\n  }\n\n        \n  fragment RelationshipAttrs on Relationship {\n    classifier\n    classifierInverse\n    annotation\n    subjectId\n    subjectName\n    createdAt\n    creatorId\n    foreignKey\n    id\n    name\n    pendingRevisions\n    relatedId\n    relatedName\n    status\n    type\n  }\n\n        \n  fragment TagAttrs on Tag {\n    category\n    createdAt\n    creatorId\n    id\n    name\n    namespace\n    pendingRevisions\n    slug\n    status\n    type\n  }\n\n        \n  fragment TaggingAttrs on Tagging {\n    annotation\n    subjectId\n    createdAt\n    creatorId\n    foreignKey\n    id\n    pendingRevisions\n    type\n    status\n    weight\n  }\n\n      ","variables":{"set":"mh2","number":"3","back":false,"moderatorView":false},"operationName":"FetchCard"}'
    ''')

    get_card_info(
        "mh2", "3", csrf_token, str(cookie_dict), debug=True
    )  # Example call to test the function

    # get_remaining_cards(
    #     get_unprocessed_cards(
    #         get_ids_of_sets_to_process(sets_to_include, bulk_file),
    #         get_id_set_already_processed(storage_file),
    #     ),
    #     X_CSRF_Token,
    #     cookie,
    #     storage_file,
    # )
