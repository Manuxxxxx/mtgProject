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


# === WORKER FUNCTION ===
def compute_synergy(args):
    a_name, a_data, b_name, b_data = args
    try:
        emb_a = torch.tensor(a_data["emb_predicted"][0])
        tag_a = torch.tensor(a_data["tags_preds_projection"][0][0])
        emb_b = torch.tensor(b_data["emb_predicted"][0])
        tag_b = torch.tensor(b_data["tags_preds_projection"][0][0])

        if emb_a.shape[-1] != 384 or emb_b.shape[-1] != 384:
            return None
        if tag_a.shape[-1] != 64 or tag_b.shape[-1] != 64:
            return None

        with torch.no_grad():
            model = ModelComplex(EMBEDDING_DIM, TAG_PROJECTOR_OUTPUT_DIM).cpu()
            model.load_state_dict(torch.load(SYNERGY_CHECKPOINT_FILE, map_location="cpu"))
            model.eval()

            logit = model(emb_a.unsqueeze(0), emb_b.unsqueeze(0), tag_b.unsqueeze(0), tag_a.unsqueeze(0))
            score = torch.sigmoid(logit).item()
            return (a_name, b_name, score)

    except Exception as e:
        print(f"Error processing {a_name} vs {b_name}: {e}")
        return None


# === MAIN SCRIPT ===
def main():
    print("🔍 Loading cards...")
    all_cards_raw = load_lookup_cards(BULK_EMBEDDING_FILE)
    all_cards = filter_cards(all_cards_raw)
    print(f"✅ Filtered {len(all_cards_raw) - len(all_cards)} cards. Remaining: {len(all_cards)}")

    card_names = list(all_cards.keys())

    print("💾 Initializing database...")
    conn = init_db()

    # Build worklist
    print("📋 Building task list...")
    to_process = []
    for i, name_a in tqdm(enumerate(card_names), total=len(card_names), desc="Processing card pairs"):
        for j in range(i + 1, len(card_names)):
            name_b = card_names[j]
            if not is_pair_done(conn, name_a, name_b):
                to_process.append((name_a, all_cards[name_a], name_b, all_cards[name_b]))

    print(f"🔢 {len(to_process)} synergy pairs to process.")

    if not to_process:
        print("🎉 All pairs are already computed.")
        return

    pool = Pool(processes=cpu_count())

    for i in tqdm(range(0, len(to_process), CHUNK_SIZE), desc="⚙️ Processing"):
        chunk = to_process[i:i + CHUNK_SIZE]
        results = pool.map(compute_synergy, chunk)

        # Save to DB
        cur = conn.cursor()
        for result in results:
            if result:
                a, b, score = result
                cur.execute("INSERT OR REPLACE INTO synergies (card_a, card_b, score) VALUES (?, ?, ?)", (a, b, score))
        conn.commit()

    pool.close()
    pool.join()
    conn.close()
    print("✅ Done! All synergy scores saved.")


if __name__ == "__main__":
    main()
