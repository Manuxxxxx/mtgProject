import os
import re
import time
import torch
import json
from mtgProject.src.models.synergy_model import build_synergy_model
from tqdm import tqdm
import random
from mtgProject.src.utils.cards_advisor import load_embeddings_cards, recommend_cards

# Load the model and predict synergies for cards that are not labeled yet.
# Saves the predicted synergies to a file, in order to label them later.

# Configuration
EMBEDDING_DIM = 384
SYNERGY_CHECKPOINT_FILE = "checkpoints/joint_training_tag/complexSin_distilbert_tag_AdamW_20250622_175623/synergy_model_epoch_9.pth"
BULK_EMBEDDING_FILE = "datasets/processed/embedding_predicted/joint_tag/cards_with_tags_20250622170831_withuri.json"
SYNERGY_FILE = "edhrec_data/labeled/with_random/random_real_synergies.json"
SYNERGY_TO_LABEL_DIR = "synergy_to_label/"
SYNERGY_MODEL_ARCH = "modelComplexSymmetrical"  # Options: "modelSimple", "modelComplex", "modelComplexSymmetrical"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_synergy(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def generate_synergy_labels(synergy_model, all_cards, all_embeddings, synergies_hashmap, synergies_per_card=10, sets_to_include=None, max_tries=100, card_to_do=100):
    synergy_model.eval()
    synergy_model.to(DEVICE)

    filteterd_cards = {}
    filtered_embeddings = {}
    for card_name, card_data in all_cards.items():
        if sets_to_include is None or card_data.get("set") in sets_to_include:
            filteterd_cards[card_name] = card_data
            filtered_embeddings[card_name] = all_embeddings[card_name]


    synergies = []
    for _ in tqdm(range(card_to_do), desc="Generating synergy labels", unit="card"):
        card_name = random.choice(list(filteterd_cards.keys()))
        
        #predict top synergies_per_card cards for the given card
        top_synergies = recommend_cards(
            [card_name],
            filtered_embeddings,
            synergy_model,
            top_n=synergies_per_card
        )
        for synergy_card_name, score in top_synergies:
            if synergy_card_name == card_name:
                continue
            synergy_key = f"{card_name}{synergy_card_name}"
            if synergy_key in synergies_hashmap:
                # If synergy already exists, skip it
                continue

            synergy_data = {
                "card1": card_name,
                "card2": synergy_card_name,
                "synergy_predicted": score
            }

            synergies.append(synergy_data)

    return synergies

            

if __name__ == "__main__":
    # Load embeddings and cards
    all_embeddings, all_cards = load_embeddings_cards(BULK_EMBEDDING_FILE)
    # Load synergy labels
    synergy_labels = load_synergy(SYNERGY_FILE)
    #create a hashmap for synergies_labels (key:card_name1card_name2, value:synergy_label)
    synergies_hashmap = {f"{synergy['card1']}{synergy['card2']}": synergy for synergy in synergy_labels}

    # Load synergy model
    synergy_model = build_synergy_model(arch_name=SYNERGY_MODEL_ARCH, embedding_dim=EMBEDDING_DIM, tag_projector_dim=EMBEDDING_DIM, initialize_weights=False)
    synergy_model.load_state_dict(torch.load(SYNERGY_CHECKPOINT_FILE, map_location=DEVICE))

    sets_to_include = [
        # 2018
        'rix', 'a25', 'dom', 'bbd', 'm19', 'grn', 'uma',
        
        # 2019
        'rna', 'war', 'mh1', 'm20', 'eld',
        
        # 2020
        'thb', 'iko', 'm21', '2xm', 'znr', 'cmr',
        
        # 2021
        'khm', 'tsr', 'stx', 'mh2', 'afr', 'mid', 'vow',
        
        # 2022
        'neo', 'snc', '2x2', 'dmu', 'bro', 'j22',
        
        # 2023
        'one', 'mom', 'mat', 'ltr', 'woe', 'lci',
        
        # 2024
        'rvr', 'mkm', 'otj', 'mh3', 'blb', 'dsk', 'fdn',
        
        # 2025 (released through May 15)
        'inr', 'dft', 'tdm'
    ]

    synergies_to_label = generate_synergy_labels(
        synergy_model,
        all_cards,
        all_embeddings,
        synergies_hashmap,
        synergies_per_card=10,
        sets_to_include=sets_to_include,  # You can specify sets to include if needed
        max_tries=10000,
        card_to_do=100,
    )

    os.makedirs(SYNERGY_TO_LABEL_DIR, exist_ok=True)
    # Save the generated synergies to a file
    with open(SYNERGY_TO_LABEL_DIR + "generated_synergies"+time.strftime("%Y%m%d%H%M%S")+".json", "w", encoding="utf-8") as f:
        json.dump(synergies_to_label, f, indent=4, ensure_ascii=False)



    

