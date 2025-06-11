import re
import torch
import json
from synergy_model import SynergyClassifier  # your binary model
from tqdm import tqdm
import random

# Configuration
EMBEDDING_DIM = 384
CHECKPOINT_FILE = "checkpoints/synergy_classifier_20250611-183940/model_epoch_36.pth"
BULK_EMBEDDING_FILE = "datasets/processed/embedding_predicted/all_commander_legal_cards20250609112722.json"
TOP_N = 30  # how many cards to recommend
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embeddings(path):
    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)
    lookup = {}
    for card in cards:
        emb = card.get("emb", [])
        if isinstance(emb, list) and len(emb) == 1 and len(emb[0]) == EMBEDDING_DIM:
            lookup[card["name"]] = torch.tensor(emb[0], dtype=torch.float)
    return lookup


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
1 Agadeem's Awakening
1 Alhammarret's Archive
1 Anguished Unmaking
1 Arcane Signet
1 Archaeomancer's Map
1 Arguel's Blood Fast
1 Auriok Champion
1 Authority of the Consuls
1 Black Market Connections
1 Blind Obedience
1 Bloodchief Ascension
1 Bojuka Bog
1 Bolas's Citadel
1 Cabal Coffers
1 Castle Locthwain
1 Caves of Koilos
1 Celestine, the Living Saint
1 Children of Korlis
1 Cleric Class
1 Command Tower
1 Crypt Ghast
1 Debt to the Deathless
1 Demonic Tutor
1 Despark
1 Erebos, God of the Dead
1 Exotic Orchard
1 Exquisite Blood
1 Farewell
1 Felidar Sovereign
1 Fellwar Stone
1 Fetid Heath
1 Final Showdown
1 Generous Gift
1 Godless Shrine
1 Gray Merchant of Asphodel
1 Heliod, Sun-Crowned
1 Heliod's Intervention
1 Insatiable Avarice
1 Isolated Chapel
1 Kambal, Consul of Allocation
1 Karlov of the Ghost Council
1 Liesa, Forgotten Archangel
1 Lightning Greaves
1 Lotho, Corrupt Shirriff
1 Necropotence
1 Orzhov Signet
1 Phyrexian Arena
1 Priest of Fell Rites
1 Reanimate
1 Reliquary Tower
1 Resplendent Angel
1 Rhox Faithmender
1 Rodolf Duskbringer
1 Sanguine Bond
1 Scoured Barrens
1 Seraph Sanctuary
1 Serra Paragon
1 Shadowspear
1 Shattered Sanctum
1 Silent Clearing
1 Smothering Tithe
1 Sol Ring
1 Solemn Simulacrum
1 Sorin Markov
1 Soul Warden
1 Soul's Attendant
1 Suture Priest
1 Swords to Plowshares
1 Tainted Field
1 Tainted Sigil
1 Takenuma, Abandoned Mire
1 Talisman of Hierarchy
1 Teferi's Protection
1 Test of Endurance
1 The Book of Exalted Deeds
1 Toxic Deluge
1 Urborg, Tomb of Yawgmoth
1 Vault of Champions
1 Vault of the Archangel
1 Vilis, Broker of Blood
1 Vito, Thorn of the Dusk Rose
1 Vizkopa Guildmage
1 Well of Lost Dreams
1 Liesa, Shroud of Dusk"""
    current_deck = decklist_to_array(deck)
    #get 30 cards from the deck at random
    if len(current_deck) > 30:
        current_deck = random.sample(current_deck, 30)

    print("deck length:", len(current_deck))

    print(" Loading data...")
    all_embeddings = load_embeddings(BULK_EMBEDDING_FILE)

    print(" Loading model...")
    model = SynergyClassifier(EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_FILE))
    model.eval()

    print(" Recommending cards...")
    top_recommendations = recommend_cards(current_deck, all_embeddings, model)

    print("\n Top Recommendations:")
    for name, score in top_recommendations:
        print(f"{name}: {score:.3f}")

    for name in current_deck:
        print(name)
    for name, _ in top_recommendations :
        print(name)