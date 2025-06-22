import re
import torch
import json
from synergy_model import ModelComplex  # your binary model
from tqdm import tqdm
import random

# Configuration
EMBEDDING_DIM = 384
CHECKPOINT_FILE = "checkpoints/joint_training/complex_noTrainBertMini_AdamW_20250613_134003/synergy_model_epoch_5.pth"
BULK_EMBEDDING_FILE = "datasets/processed/embedding_predicted/joint/commander_legal_cards20250609112722.json"
TOP_N = 30  # how many cards to recommend
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embeddings_cards(path):
    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)
    lookup_emb = {}
    lookup_cards = {}
    for card in cards:
        emb = card.get("emb", [])
        if isinstance(emb, list) and len(emb) == 1 and len(emb[0]) == EMBEDDING_DIM:
            lookup_emb[card["name"]] = torch.tensor(emb[0], dtype=torch.float)
        
        lookup_cards[card["name"]] = card
    return lookup_emb, lookup_cards


def recommend_cards(current_deck, all_embeddings, model, top_n=TOP_N):
    model.eval()
    candidates = []

    deck_embeddings = [all_embeddings[name] for name in current_deck if name in all_embeddings]
    if not deck_embeddings:
        raise ValueError("No valid embeddings found for current deck.")

    with torch.no_grad():
        for card_name, emb in tqdm(all_embeddings.items(), desc="Evaluating cards", unit="card"):
            if card_name in current_deck:
                continue  # skip already selected cards

            scores = []
            for deck_emb in deck_embeddings:
                input_emb = torch.cat([deck_emb.unsqueeze(0), emb.unsqueeze(0)], dim=1).to(DEVICE)
                logit = model(input_emb[:, :EMBEDDING_DIM], input_emb[:, EMBEDDING_DIM:])
                score = torch.sigmoid(logit).item()
                scores.append(score)

            avg_score = sum(scores) / len(scores)
            candidates.append((card_name, avg_score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]

def recommend_cards_commander(current_deck, all_embeddings, all_cards, model, top_n=TOP_N, commander_name=None):
    model.eval()
    candidates = []

    # Validate current deck embeddings
    deck_embeddings = [all_embeddings[name] for name in current_deck if name in all_embeddings]
    if not deck_embeddings:
        raise ValueError("No valid embeddings found for current deck.")

    # Commander colors (if given)
    allowed_colors = None
    if commander_name:
        commander_info = all_cards.get(commander_name)
        if commander_info and "colors" in commander_info:
            allowed_colors = set(commander_info["colors"])
        else:
            raise ValueError(f"Commander {commander_name} not found or has no color info.")

    with torch.no_grad():
        for card_name, emb in tqdm(all_embeddings.items(), desc="Evaluating cards", unit="card"):
            if card_name in current_deck:
                continue  # Skip existing cards

            card_info = all_cards.get(card_name, {})
            if allowed_colors:
                card_colors = set(card_info.get("colors", []))
                if not card_colors.issubset(allowed_colors):
                    continue  # Skip cards outside commander's color identity

            # Synergy score between candidate and each card in the deck
            scores = []
            for deck_emb in deck_embeddings:
                input_emb = torch.cat([deck_emb.unsqueeze(0), emb.unsqueeze(0)], dim=1).to(DEVICE)
                logit = model(input_emb[:, :EMBEDDING_DIM], input_emb[:, EMBEDDING_DIM:])
                # print(f"Evaluating {card_name} against deck embeddings. logit shape: {logit.shape}, logit: {logit}")
                score = torch.sigmoid(logit).item()
                scores.append(score)

            # Weighted score: prefer cards that synergize broadly across deck
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            final_score = 0.7 * avg_score + 0.3 * max_score  # tunable weights

            candidates.append((card_name, final_score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]


def decklist_to_array(decklist):
    """
    Convert a decklist string into a list of card names.
    Each line is expected to start with quantity, followed by card name (with optional set/code info).
    """
    lines = decklist.strip().split('\n')
    card_names = []

    for line in lines:
        # Remove quantity and extract name before set info
        match = re.match(r'^\d+\s+(.*?)(?:\s+\([A-Z]+\).*|\s+\w+-\d+)?$', line)
        if match:
            name = match.group(1).strip()
            card_names.append(name)

    return card_names


if __name__ == "__main__":
    deck = """1 Aetherflux Reservoir
1 Authority of the Consuls
1 Black Market Connections
1 Blind Obedience
1 Bloodchief Ascension
1 Cleric Class
1 Liesa, Shroud of Dusk
"""
    current_deck = decklist_to_array(deck)
    commander = "Liesa, Shroud of Dusk"
    #get 30 cards from the deck at random
    if len(current_deck) > 30:
        current_deck = random.sample(current_deck, 30)

    print("deck length:", len(current_deck))

    print(" Loading data...")
    all_embeddings, all_cards = load_embeddings_cards(BULK_EMBEDDING_FILE)

    print(" Loading model...")
    model = ModelComplex(EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_FILE))
    model.eval()

    # print(" Recommending cards...")
    # top_recommendations = recommend_cards(current_deck, all_embeddings, model)

    # print("\n Top Recommendations:")
    # for name, score in top_recommendations:
    #     print(f"{name}: {score:.3f}")

    # for name in current_deck:
    #     print(name)
    # for name, _ in top_recommendations :
    #     print(name)

    print("\n Recommending cards with commander...")
    top_recommendations_commander = recommend_cards_commander(
        current_deck, all_embeddings, all_cards, model, commander_name=commander
    )
    print("\n Top Recommendations with Commander:")
    for name, score in top_recommendations_commander:
        print(f"{name}: {score:.3f}")
    for name in current_deck:
        print(name)
    for name, _ in top_recommendations_commander:
        print(name)