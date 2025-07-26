import re
import torch
import json
from mtgProject.src.models.synergy_model import ModelComplex  # your binary model
from mtgProject.src.models.tag_model import TagModel  # your tag model
from tqdm import tqdm
import random

# Configuration
EMBEDDING_DIM = 384
TAG_PROJECTOR_OUTPUT_DIM = 64
SYNERGY_CHECKPOINT_FILE = "checkpoints/two_phase_joint/two_phase_joint_training_tag_20250717_122452/synergy_model_epoch_18.pth"
BULK_EMBEDDING_FILE = "datasets/processed/embedding_predicted/joint_tag/cards_with_tags_20250718003320.json"
TOP_N = 40  # how many cards to recommend
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lookup_cards(path):
    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)
    lookup_cards = {}
    for card in cards:        
        lookup_cards[card["name"]] = card
    return lookup_cards


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

def recommend_cards_commander(current_deck, all_cards, model, top_n=TOP_N, commander_name=None):
    model.eval()
    candidates = []

    # Commander color identity (if provided)
    allowed_colors = None
    if commander_name:
        commander_info = all_cards.get(commander_name)
        if commander_info and "colors" in commander_info:
            allowed_colors = set(commander_info["colors"])
        else:
            raise ValueError(f"Commander '{commander_name}' not found or missing color info.")

    # Preprocess deck embeddings and tag projections
    deck_embs, deck_tags = [], []

    for deck_card_name in current_deck:
        deck_card = all_cards.get(deck_card_name)
        if not deck_card:
            continue

        try:
            emb = torch.tensor(deck_card["emb_predicted"][0], device=DEVICE)
            tag = torch.tensor(deck_card["tags_preds_projection"][0][0], device=DEVICE)
        except (KeyError, IndexError, TypeError):
            print(f"Skipping malformed deck card: {deck_card_name}")
            continue

        if emb.shape[-1] != 384 or tag.shape[-1] != 64:
            print(f"Skipping {deck_card_name} due to invalid shape: emb={emb.shape}, tag={tag.shape}")
            continue

        deck_embs.append(emb)
        deck_tags.append(tag)

    if not deck_embs:
        raise ValueError("No valid deck embeddings found.")

    deck_embs_tensor = torch.stack(deck_embs)       # shape: [deck_size, 384]
    deck_tags_tensor = torch.stack(deck_tags)       # shape: [deck_size, 64]

    with torch.no_grad():
        for card_name, card in tqdm(all_cards.items(), desc="Evaluating cards", unit="card"):
            if card_name in current_deck:
                continue

            if allowed_colors:
                card_colors = set(card.get("colors", []))
                if not card_colors.issubset(allowed_colors):
                    continue

            try:
                emb = torch.tensor(card["emb_predicted"][0], device=DEVICE)
                tag_proj = torch.tensor(card["tags_preds_projection"][0][0], device=DEVICE)
            except (KeyError, IndexError, TypeError):
                continue

            if emb.shape[-1] != 384 or tag_proj.shape[-1] != 64:
                continue

            # Expand candidate to match deck size for batch inference
            batch_emb = emb.unsqueeze(0).expand(deck_embs_tensor.size(0), -1)         # [deck_size, 384]
            batch_tag_proj = tag_proj.unsqueeze(0).expand(deck_tags_tensor.size(0), -1)  # [deck_size, 64]

            try:
                logits = model(batch_emb, deck_embs_tensor, deck_tags_tensor, batch_tag_proj)
                if logits is None:
                    continue

                scores = torch.sigmoid(logits).squeeze()
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)  # Handle scalar case
                scores = scores.tolist()

                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                final_score = 0.7 * avg_score + 0.3 * max_score
                candidates.append((card_name, final_score))

            except Exception as e:
                print(f"Error processing {card_name}: {e}")
                continue

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

# def print_tags_from_tag_model(tag_model, card_name, all_cards, all_embeddings):
#     """
#     Print tags for a given card using the tag model.
#     """
#     if card_name not in all_embeddings:
#         print(f"Card '{card_name}' not found in embeddings.")
#         return

#     emb = all_embeddings[card_name].unsqueeze(0).to(DEVICE)
#     with torch.no_grad():




if __name__ == "__main__":
    deck = """
1 Cinder Glade
1 Clifftop Retreat
1 Command Tower
1 Cultivate
1 Dawn's Truce
1 Dazzling Theater // Prop Room
1 Doubling Season
1 Drumbellower
1 Elspeth, Storm Slayer
1 Enduring Vitality
1 Exotic Orchard
1 Farmer Cotton
1 Farseek
1 Finneas, Ace Archer
1 For the Common Good
1 Gilded Goose
1 Grand Crescendo
1 Halo Fountain
1 Hare Apparent
1 Heroic Intervention
1 Hop to It
1 Hour of Reckoning
1 Idol of Oblivion
1 Impact Tremors
1 Intangible Virtue
1 Jacked Rabbit
1 Jaheira, Friend of the Forest
1 Jetmir's Garden
1 Jetmir, Nexus of Revels
1 Jungle Shrine
1 Lightning Greaves
1 March of the Multitudes
1 Mondrak, Glory Dominus
1 Nature's Lore
1 Ocelot Pride
1 Ojer Taq, Deepest Foundation
1 Parallel Lives
1 Path to Exile
1 Peregrin Took
1 Purphoros, God of the Forge
1 Queen Allenal of Ruadach
1 Rampant Growth
1 Reliquary Tower
1 Rhys the Redeemed
1 Rootbound Crag
1 Rosie Cotton of South Lane
1 Sacred Foundry
1 Scute Swarm
1 Season of the Burrow
1 Second Harvest
1 Secure the Wastes
1 Seedborn Muse
1 Skullclamp
1 Smothering Tithe
"""
    current_deck = decklist_to_array(deck)
    commander = None
    #get 30 cards from the deck at random
    if len(current_deck) > 30:
        current_deck = random.sample(current_deck, 30)

    print("deck length:", len(current_deck))

    print(" Loading data...")
    all_cards = load_lookup_cards(BULK_EMBEDDING_FILE)

    print(" Loading model...")
    synergy_model = ModelComplex(EMBEDDING_DIM, TAG_PROJECTOR_OUTPUT_DIM).to(DEVICE)
    synergy_model.load_state_dict(torch.load(SYNERGY_CHECKPOINT_FILE))
    synergy_model.eval()


    # print(" Recommending cards...")
    # top_recommendations = recommend_cards(current_deck, all_embeddings, synergy_model)

    # print("\n Top Recommendations:")
    # for name, score in top_recommendations:
    #     print(f"{name}: {score:.3f}")

    # for name in current_deck:
    #     print(name)
    # for name, _ in top_recommendations :
    #     print(name)

    print("\n Recommending cards with commander...")
    top_recommendations_commander = recommend_cards_commander(
        current_deck, all_cards, synergy_model, commander_name=commander
    )
    print("\n Top Recommendations with Commander:")
    for name, score in top_recommendations_commander:
        print(f"{name}: {score:.3f}")
    for name in current_deck:
        print(name)
    for name, _ in top_recommendations_commander:
        print(name)