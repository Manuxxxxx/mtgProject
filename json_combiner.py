import json
from typing import Counter
import conf
import os
from datetime import datetime

def get_card_name_from_set_number(set_code, collector_number, data):
    
    # Iterate through the cards in the JSON data
    for card in data:
        if card.get('set_code') == set_code and card.get('collector_number') == collector_number:
            return card
    
    return None

def get_tags_from_set_number(set_code, collector_number, tag_data):
    #get all the tags for a specific set code and collector number
    tags = []
    for set_code_key, data in tag_data.items():
        if set_code_key == set_code:
            for key, value in data.items():
                if key == "collector_number":
                    continue
                for tag_entry in value:
                    if tag_entry.get("collector_number") == collector_number:
                        tags.append(tag_entry["tag"])
    return tags

def get_all_card_without_duplicate(input_file_processed_json):
    with open(input_file_processed_json, 'r') as file:
        data = json.load(file)

    print(f"Total cards before removing duplicates: {len(data)}")
    # remove duplicates -> same name
    unique_cards = {}
    for card in data:
        name = card.get('name')
        if name not in unique_cards:
            unique_cards[name] = card
    print(f"Total cards after removing duplicates: {len(unique_cards)}")
    return list(unique_cards.values())

def add_tag_to_card(card, tag_data):
    set_code = card.get('set')
    collector_number = card.get('collector_number')

    # Get the tag from the tag data
    tags = get_tags_from_set_number(set_code, collector_number, tag_data)
    if tags:
        card['tags'] = tags
    else:
        card['tags'] = None
    return card

def add_tags_to_all_cards(card_data, tag_data):
    # Add tags to all cards
    for card in card_data:
        card = add_tag_to_card(card, tag_data)
    
    return card_data

def extract_all_tags(tag_data):

    unique_tags = set()

    for set_code, data in tag_data.items():
        for key, value in data.items():
            if key == "collector_number":
                continue
            for tag_entry in value:
                tag = tag_entry.get("tag")
                if tag:
                    unique_tags.add(tag)

    return list(unique_tags)


def extract_all_tags_with_min_freq(tag_data, min_count=20):
    tag_counter = Counter()

    for set_code, data in tag_data.items():
        for key, value in data.items():
            if key == "collector_number":
                continue
            for tag_entry in value:
                tag_counter[tag_entry["tag"]] += 1

    return {tag for tag, count in tag_counter.items() if count >= min_count}

def filter_cards_by_sets(card_data, sets_to_include):
    filtered_cards = []
    for card in card_data:
        if card.get('set') in sets_to_include:
            filtered_cards.append(card)
    return filtered_cards


if __name__ == "__main__":
    input_file_tagger = conf.scrapped_store
    tag_data = json.load(open(input_file_tagger, 'r'))

    input_file_processed_json = conf.processed_json

    card_data = filter_cards_by_sets(get_all_card_without_duplicate(input_file_processed_json), conf.sets_to_include)
    print(f"Total cards after set filter: {len(card_data)}")
    card_data = add_tags_to_all_cards(card_data, tag_data)
    # Save the updated card data to a new JSON file
    output_file = os.path.join(conf.processed_tag_dir, f"cards_with_tags_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(card_data, f, indent=2)
    

    # all_tags = extract_all_tags(tag_data)
    # print(f"Total tags: {len(all_tags)}")


