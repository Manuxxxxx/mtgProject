import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ----------------------
# Configuration
# ----------------------
TAG_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" #Output dimension: 384
INPUT_DIR = "datasets/processed/tag_included/"
OUTPUT_DIR = "datasets/processed/embedding/"

INPUT_JSON = INPUT_DIR+"cards.json"
OUTPUT_JSON = OUTPUT_DIR+"cards_with_embeddings.json"

# ----------------------
# Load Embedding Model
# ----------------------
model = SentenceTransformer(TAG_EMBED_MODEL)

# ----------------------
# Load Data
# ----------------------
with open(INPUT_JSON, "r") as f:
    cards = json.load(f)

# ----------------------
# Embed Tags and Create Target Embeddings
# ----------------------
def embed_tags(tags):
    if not tags:
        return None
    tag_embeddings = model.encode(tags)
    return np.mean(tag_embeddings, axis=0)

processed_cards = []
for card in tqdm(cards, desc="Embedding cards"):
    tags = card.get("tags", [])
    if tags:
        embedding = embed_tags(tags)
        if embedding is not None:
            card["target_embedding"] = embedding.tolist()
            processed_cards.append(card)

# ----------------------
# Save Output
# ----------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(processed_cards, f, indent=2)

print(f"✅ Saved {len(processed_cards)} cards with embeddings to {OUTPUT_JSON}")
