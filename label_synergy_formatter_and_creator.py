import json
import random

# --- Config ---
LABEL_FILE = "edhrec_data/labeled/combined_commander_synergy_data.json"
BULK_FILE = "datasets/processed/embedding_predicted/all_commander_legal_cards20250609112722.json"
OUTPUT_FILE = "edhrec_data/labeled/with_random/random_real_synergies.json"
NEGATIVE_SAMPLES_PER_COMMANDER = 50  # How many random 0-synergy to add per commander
ADDITIONAL_RANDOM_NEGATIVES = 3000 

# --- Load data ---
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    synergy_data = json.load(f)

with open(BULK_FILE, "r", encoding="utf-8") as f:
    all_cards = json.load(f)

all_card_names = sorted(set(card["name"] for card in all_cards))

# --- Process ---
flattened = []

all_pairs_set = set()
for entry in synergy_data:
    commander = entry["commander"]

    true_synergies = set()
    
    # Add existing synergy=1 labels
    for label in entry["labels"]:
        card_name = label["card2"]["name"]
        flattened.append({
            "card1": {"name": commander},
            "card2": {"name": card_name},
            "synergy": label["synergy"],
            "synergy_edhrec": label["card2"]["synergy"]
        })
        true_synergies.add(card_name)
        all_pairs_set.add(tuple(sorted([commander, card_name])))

    # Add synthetic synergy=0 negatives
    available_negatives = list(set(all_card_names) - true_synergies - {commander})
    sampled_negatives = random.sample(available_negatives, min(NEGATIVE_SAMPLES_PER_COMMANDER, len(available_negatives)))

    for neg_card in sampled_negatives:
        flattened.append({
            "card1": {"name": commander},
            "card2": {"name": neg_card},
            "synergy": 0
        })
        all_pairs_set.add(tuple(sorted([commander, neg_card])))

print(f" Adding {ADDITIONAL_RANDOM_NEGATIVES} random 0-synergy pairs...")
added_randoms = 0
tries = 0
max_tries = 10  # avoid infinite loop

while added_randoms < ADDITIONAL_RANDOM_NEGATIVES and tries < max_tries:
    card1, card2 = random.sample(all_card_names, 2)
    pair_key = tuple(sorted([card1, card2]))
    if pair_key not in all_pairs_set:
        flattened.append({
            "card1": {"name": card1},
            "card2": {"name": card2},
            "synergy": 0
        })
        all_pairs_set.add(pair_key)
        added_randoms += 1
        tries = 0  # reset tries after a successful addition
    else: 
        tries += 1


# --- Save ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(flattened, f, indent=2)

print(f"✅ Flattened synergy file saved to: {OUTPUT_FILE}")
print(f"🔍 Unique pairs: {len(all_pairs_set)}")
