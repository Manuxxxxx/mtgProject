import json
import torch
import sqlite3
from tqdm import tqdm
from math import comb

from src.models.synergy_model import (
    ModelComplex,
)  # Replace with actual import path to your model
from src.utils.cards_advisor import load_lookup_cards

# === CONFIG ===
EMBEDDING_DIM = 384
TAG_PROJECTOR_OUTPUT_DIM = 64
SYNERGY_CHECKPOINT_FILE = "checkpoints/two_phase_joint/two_phase_joint_training_tag_20250717_122452/synergy_model_epoch_18.pth"
BULK_EMBEDDING_FILE = "datasets/processed/embedding_predicted/joint_tag/cards_with_tags_20250718003320.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DB_FILE = "synergy_cache_compressed.sqlite"
CHUNK_SIZE = 15000
# ==============


def filter_cards(all_cards):
    filtered = {}
    for name, card in all_cards.items():
        if "Land" in card.get("type_line", ""):
            continue
        filtered[name] = card
    return filtered


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS synergies (
            idx_a INTEGER,
            idx_b INTEGER,
            score REAL,
            PRIMARY KEY (idx_a, idx_b)
        );
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value INTEGER
        );
    """
    )
    conn.commit()
    return conn


def get_last_processed_index(conn):
    cur = conn.cursor()
    cur.execute("SELECT value FROM meta WHERE key = 'last_idx_a'")
    row = cur.fetchone()
    return int(row[0]) if row else 0


def set_last_processed_index(conn, idx):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("last_idx_a", idx)
    )
    conn.commit()


def batch_pairs(card_count, chunk_size, start_idx=0):
    batch = []
    for i in range(start_idx, card_count):
        for j in range(i + 1, card_count):
            batch.append((i, j))
            if len(batch) >= chunk_size:
                yield batch
                batch = []
    if batch:
        yield batch


def count_remaining_pairs(n, start_idx):
    return comb(n - start_idx, 2)


def main():
    debug = False

    print("🔍 Loading and filtering cards...")
    all_cards_raw = load_lookup_cards(BULK_EMBEDDING_FILE)
    all_cards = filter_cards(all_cards_raw)
    print(
        f"✅ Filtered {len(all_cards_raw) - len(all_cards)} cards. Remaining: {len(all_cards)}"
    )

    card_names = list(all_cards.keys())
    name_to_idx = {name: i for i, name in enumerate(card_names)}
    idx_to_name = {i: name for i, name in enumerate(card_names)}
    n = len(card_names)
    total_possible_pairs = n * (n - 1) // 2

    print("💾 Initializing database...")
    conn = init_db()
    cur = conn.cursor()
    last_idx = get_last_processed_index(conn)

    print("📦 Loading model...")
    model = ModelComplex(EMBEDDING_DIM, TAG_PROJECTOR_OUTPUT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(SYNERGY_CHECKPOINT_FILE, map_location=DEVICE))
    model.eval()

    print("⚙️ Caching embeddings on GPU...")
    emb_dict = {
        i: torch.tensor(all_cards[name]["emb_predicted"][0], device=DEVICE)
        for i, name in idx_to_name.items()
    }
    tag_dict = {
        i: torch.tensor(all_cards[name]["tags_preds_projection"][0][0], device=DEVICE)
        for i, name in idx_to_name.items()
    }

    remaining_pairs = count_remaining_pairs(n, last_idx)
    remaining_batches = remaining_pairs // CHUNK_SIZE + (
        1 if remaining_pairs % CHUNK_SIZE > 0 else 0
    )
    done_pairs = total_possible_pairs - remaining_pairs
    session_pairs = 0  # Pairs processed during this run

    print(f"📊 Estimated progress:")
    print(f"   ✅ Done: {done_pairs:,}")
    print(f"   🕐 Remaining: {remaining_pairs:,} of {total_possible_pairs:,} total")

    if debug:
        actual = cur.execute("SELECT COUNT(*) FROM synergies").fetchone()[0]
        print(f"   ⚙️ Actual entries in DB: {actual}")

    print("🚀 Starting batch processing...")

    current_i = last_idx

    for batch in tqdm(
        batch_pairs(n, CHUNK_SIZE, start_idx=last_idx),
        desc="Processing pairs",
        total=remaining_batches,
    ):
        emb_a_batch = torch.stack([emb_dict[i] for i, j in batch])
        tag_a_batch = torch.stack([tag_dict[i] for i, j in batch])
        emb_b_batch = torch.stack([emb_dict[j] for i, j in batch])
        tag_b_batch = torch.stack([tag_dict[j] for i, j in batch])

        with torch.no_grad():
            logits = model(emb_a_batch, emb_b_batch, tag_b_batch, tag_a_batch)
            scores = torch.sigmoid(logits).squeeze().cpu().tolist()

        for (i, j), score in zip(batch, scores):
            cur.execute(
                "INSERT OR REPLACE INTO synergies (idx_a, idx_b, score) VALUES (?, ?, ?)",
                (i, j, score),
            )
        conn.commit()

        # Checkpoint after batch (based on first i in batch)
        current_i = batch[0][0]
        set_last_processed_index(conn, current_i)

        done_pairs += len(batch)
        session_pairs += len(batch)
        remaining_pairs = total_possible_pairs - done_pairs

        # Print progress every 10 million
        if session_pairs % 10_000_000 < len(batch):
            print("\n------ Progress Update ------")
            print(f"\n📈 Progress Update:")
            print(f"   ✅ Total processed: {done_pairs:,}")
            print(f"   🔄 Session processed: {session_pairs:,}")
            print(f"   ⏳ Remaining: {remaining_pairs:,}")
            print(f"   📊 Total: {total_possible_pairs:,}\n")

    conn.close()
    print("\n------------------------------------")
    print("✅ All pairs processed and saved to database.")
    print(f"\n📈 Progress Update:")
    print(f"   ✅ Total processed: {done_pairs:,}")
    print(f"   🔄 Session processed: {session_pairs:,}")
    print(f"   ⏳ Remaining: {remaining_pairs:,}")
    print(f"   📊 Total: {total_possible_pairs:,}\n")


if __name__ == "__main__":
    main()
