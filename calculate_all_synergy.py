import json
import torch
import sqlite3
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from synergy_model import ModelComplex  # Replace with actual import path to your model
from cards_advisor import load_lookup_cards

# === CONFIG ===
EMBEDDING_DIM = 384
TAG_PROJECTOR_OUTPUT_DIM = 64
SYNERGY_CHECKPOINT_FILE = "checkpoints/two_phase_joint/two_phase_joint_training_tag_20250717_122452/synergy_model_epoch_18.pth"
BULK_EMBEDDING_FILE = "datasets/processed/embedding_predicted/joint_tag/cards_with_tags_20250718003320.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DB_FILE = "synergy_cache.sqlite"
CHUNK_SIZE = 1000
# ==============


# === FILTERING FUNCTION ===
def filter_cards(all_cards):
    """
    Filters out cards based on configurable conditions.
    Modify or comment out filters as needed.
    """
    filtered = {}
    for name, card in all_cards.items():
        type_line = card.get("type_line", "")

        # ❌ Ignore Lands
        if "Land" in type_line:
            continue

        # ✅ Other filters (optional - toggle by uncommenting)

        # Ignore cards with missing embeddings
        # if "emb_predicted" not in card or "tags_preds_projection" not in card:
        #     continue

        # Ignore tokens
        # if "Token" in type_line:
        #     continue

        # Ignore silver-border cards
        # if card.get("border_color") == "silver":
        #     continue

        filtered[name] = card
    return filtered


# === SQLITE DB SETUP ===
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS synergies (
            card_a TEXT,
            card_b TEXT,
            score REAL,
            PRIMARY KEY (card_a, card_b)
        );
    """)
    conn.commit()
    return conn


def is_pair_done(conn, a, b):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM synergies WHERE card_a=? AND card_b=?", (a, b))
    return cur.fetchone() is not None


def get_last_processed_card(conn):
    cur = conn.cursor()
    cur.execute("SELECT card_a FROM synergies ORDER BY rowid DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else None

def batch_pairs(card_names, chunk_size, start_from=None):
    """
    Generator yielding batches of unique (a, b) pairs.
    """
    start_index = 0
    if start_from:
        try:
            start_index = card_names.index(start_from)
        except ValueError:
            print(f"⚠️ Warning: start_from '{start_from}' not found in card_names")

    total = len(card_names)
    batch = []
    for i in range(start_index, total):
        a = card_names[i]
        for j in range(i + 1, total):
            b = card_names[j]
            batch.append((a, b))
            if len(batch) >= chunk_size:
                yield batch
                batch = []
    if batch:
        yield batch


def filter_existing_pairs(conn, pairs):
    """
    Query DB to find which pairs are already computed.
    Returns filtered pairs that are NOT in DB yet.
    """
    if not pairs:
        return []

    placeholders = ",".join(["(?,?)"] * len(pairs))
    query = f"SELECT card_a, card_b FROM synergies WHERE (card_a, card_b) IN ({placeholders})"
    flattened = []
    for a, b in pairs:
        flattened.extend([a, b])

    cur = conn.cursor()
    cur.execute(query, flattened)
    done_pairs = set(cur.fetchall())

    filtered = [p for p in pairs if p not in done_pairs]
    return filtered


def main():
    print("🔍 Loading cards...")
    all_cards_raw = load_lookup_cards(BULK_EMBEDDING_FILE)
    all_cards = filter_cards(all_cards_raw)
    print(f"✅ Filtered {len(all_cards_raw) - len(all_cards)} cards. Remaining: {len(all_cards)}")

    card_names = list(all_cards.keys())

    print("💾 Initializing database...")
    conn = init_db()

    print("📦 Loading model to GPU...")
    model = ModelComplex(EMBEDDING_DIM, TAG_PROJECTOR_OUTPUT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(SYNERGY_CHECKPOINT_FILE, map_location=DEVICE))
    model.eval()

    print("📥 Preloading card tensors on GPU...")
    emb_dict = {}
    tag_dict = {}
    for name, card in all_cards.items():
        emb = torch.tensor(card["emb_predicted"][0], device=DEVICE)
        tag = torch.tensor(card["tags_preds_projection"][0][0], device=DEVICE)
        emb_dict[name] = emb
        tag_dict[name] = tag

    print("📍 Getting last processed card to resume...")
    last_processed_card = get_last_processed_card(conn)
    print(f"🔁 Resuming from card: {last_processed_card}")

    cur = conn.cursor()
    total_pairs_processed = 0

    # Lazy batch pairs generator
    print("⚙️ Processing synergy pairs in batches...")
    for pairs_batch in tqdm(
        batch_pairs(card_names, CHUNK_SIZE, start_from=last_processed_card),
        desc="Processing pairs",
        total=len(card_names) * (len(card_names) - 1) // 2 // CHUNK_SIZE,
    ):
        pairs_batch = filter_existing_pairs(conn, pairs_batch)
        if not pairs_batch:
            continue

        emb_a_batch = torch.stack([emb_dict[a] for a, b in pairs_batch]).to(DEVICE)
        tag_a_batch = torch.stack([tag_dict[a] for a, b in pairs_batch]).to(DEVICE)
        emb_b_batch = torch.stack([emb_dict[b] for a, b in pairs_batch]).to(DEVICE)
        tag_b_batch = torch.stack([tag_dict[b] for a, b in pairs_batch]).to(DEVICE)

        with torch.no_grad():
            logits = model(emb_a_batch, emb_b_batch, tag_b_batch, tag_a_batch)
            scores = torch.sigmoid(logits).squeeze().cpu().tolist()

        for (a, b), score in zip(pairs_batch, scores):
            cur.execute(
                "INSERT OR REPLACE INTO synergies (card_a, card_b, score) VALUES (?, ?, ?)",
                (a, b, score),
            )
        conn.commit()
        total_pairs_processed += len(pairs_batch)

    conn.close()
    print(f"✅ Done! Processed {total_pairs_processed} synergy pairs.")



if __name__ == "__main__":
    main()
