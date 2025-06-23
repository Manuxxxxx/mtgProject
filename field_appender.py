import json
from typing import Counter
import conf
import os
from datetime import datetime


def append_field_from_bulk_to_processed(bulk_file, processed_file, fields_to_append=None):
    """
    Append fields from bulk_file to processed_file based on card names.
    If fields_to_append is None, all fields from bulk_file will be appended.
    """
    with open(bulk_file, 'r') as bulk_f:
        bulk_data = json.load(bulk_f)

    with open(processed_file, 'r') as processed_f:
        processed_data = json.load(processed_f)

    # Create a mapping of card names to their data in the bulk file
    bulk_mapping = {card['name']: card for card in bulk_data}

    for card in processed_data:
        name = card.get('name')
        if name in bulk_mapping:
            if fields_to_append is not None:
                # Append only specified fields
                for field in fields_to_append:
                    if field in bulk_mapping[name]:
                        card[field] = bulk_mapping[name][field]

    # Save the updated processed data back to the file
    with open(processed_file, 'w') as processed_f:
        json.dump(processed_data, processed_f, indent=4)

if __name__ == "__main__":
    bulk_file = "datasets/bulk/oracle-cards-20250513210734.json"
    processed_file = "datasets/processed/tag_included/cards_with_tags_20250622170831_copy.json"

    fields_to_append = ["image_uris"]

    append_field_from_bulk_to_processed(bulk_file, processed_file, fields_to_append)

    