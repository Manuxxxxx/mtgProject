import json
import torch
import sqlite3
from tqdm import tqdm
from math import comb

# Replace direct class import with factory
from src.models.synergy_model import build_synergy_model
from src.utils.cards_advisor import load_lookup_cards
from src.models.tag_model import build_tag_model
from src.models.tag_projector_model import build_tag_projector_model
from src.synergy_navigation.embeddings_calculator import safe_load_tag_model  # reuse robust loader

# === CONFIG ===
EMBEDDING_DIM = 384
# Synergy model selection: use hidden tag state OR projector output
USE_TAG_PROJECTOR = False  # If True use projector vectors, else use tag model hidden states
SYNERGY_ARCH = "modelComplexTagHidden"  # Must match representation type (modelComplex / modelComplexSymmetrical for projector, *TagHidden* for hidden)
MIXED_PRECISION = True

# Synergy checkpoint
SYNERGY_CHECKPOINT_FILE = "checkpoints/two_phase_joint/two_phase_joint__tag_hidden__detach_20250822_150413/synergy_model_epoch_44.pth"

# Input cards file (only embeddings + metadata needed; tags inside file will be ignored)
BULK_EMBEDDING_FILE = "datasets/processed/embedding_predicted/joint_tag/cards_with_tags_174_20250822234013.json"

# Tag model (must align with training used for synergy model)
TAG_MODEL_ARCH = "simple"
TAG_MODEL_HIDDEN_DIMS = [512, 512]
TAG_MODEL_OUTPUT_DIM = 174  # number of tags (logits/probs output dim)
TAG_MODEL_DROPOUT = 0.3
TAG_MODEL_USE_SIGMOID_OUTPUT = True  # we want probabilities when projecting
TAG_MODEL_CHECKPOINT_FILE = "checkpoints/two_phase_joint/two_phase_joint__tag_hidden__detach_20250822_150413/tag_multi_model_epoch_44.pth"
TAG_LAST_HIDDEN_DIM = TAG_MODEL_HIDDEN_DIMS[-1]  # hidden state size passed to synergy when not using projector

# Optional Tag projector
TAG_PROJECTOR_CHECKPOINT_FILE = None  # e.g. "checkpoints/.../tag_projector_model_epoch_X.pth"
TAG_PROJECTOR_OUTPUT_DIM = 64  # dimension of tag projector vectors if used
TAG_PROJECTOR_HIDDEN_DIM = None  # if your projector has hidden layer; set if needed
TAG_PROJECTOR_DROPOUT = 0.3

# Performance
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_TAG_INFERENCE = 256

# DB / batching
DB_FILE = "synergy_cache_compressed_174_20250822234013.sqlite"
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


def load_models():
    # Build synergy model according to representation type
    synergy_model = build_synergy_model(
        arch_name=SYNERGY_ARCH,
        embedding_dim=EMBEDDING_DIM,
        tag_projector_dim=TAG_PROJECTOR_OUTPUT_DIM if USE_TAG_PROJECTOR else None,
        initialize_weights=False,
        hidden_tag_dim=TAG_LAST_HIDDEN_DIM if not USE_TAG_PROJECTOR else None,
    ).to(DEVICE)
    synergy_model.load_state_dict(torch.load(SYNERGY_CHECKPOINT_FILE, map_location=DEVICE))
    synergy_model.eval()

    # Tag model
    tag_model = build_tag_model(
        arch_name=TAG_MODEL_ARCH,
        input_dim=EMBEDDING_DIM,
        hidden_dims=TAG_MODEL_HIDDEN_DIMS,
        output_dim=TAG_MODEL_OUTPUT_DIM,
        dropout=TAG_MODEL_DROPOUT,
        use_batchnorm=False,
        use_sigmoid_output=TAG_MODEL_USE_SIGMOID_OUTPUT,
    ).to(DEVICE)
    tag_model = safe_load_tag_model(tag_model, TAG_MODEL_CHECKPOINT_FILE, DEVICE)
    tag_model.eval()

    # Optional projector
    tag_projector_model = None
    if USE_TAG_PROJECTOR:
        tag_projector_model = build_tag_projector_model(
            num_tags=TAG_MODEL_OUTPUT_DIM,
            hidden_dim=TAG_PROJECTOR_HIDDEN_DIM,
            output_dim=TAG_PROJECTOR_OUTPUT_DIM,
            dropout=TAG_PROJECTOR_DROPOUT,
        ).to(DEVICE)
        if TAG_PROJECTOR_CHECKPOINT_FILE:
            tag_projector_model.load_state_dict(
                torch.load(TAG_PROJECTOR_CHECKPOINT_FILE, map_location=DEVICE)
            )
        tag_projector_model.eval()

    return synergy_model, tag_model, tag_projector_model


def extract_embeddings(all_cards, idx_to_name):
    emb_dict = {}
    for i, name in idx_to_name.items():
        card = all_cards[name]
        emb_raw = card.get("emb_predicted")
        if isinstance(emb_raw, list):
            if len(emb_raw) > 0 and isinstance(emb_raw[0], list):
                emb_vec = emb_raw[0]
            else:
                emb_vec = emb_raw
        else:
            raise ValueError(f"Unexpected emb_predicted format for card {name}")
        emb_dict[i] = torch.tensor(emb_vec, device=DEVICE, dtype=torch.float32)
    return emb_dict


def compute_tag_representations(emb_dict):
    # Produce a dictionary idx -> tag vector (hidden or projected)
    indices = list(emb_dict.keys())
    vectors = {}

    # Prepare batch tensors
    with torch.no_grad():
        for k in range(0, len(indices), BATCH_SIZE_TAG_INFERENCE):
            batch_indices = indices[k : k + BATCH_SIZE_TAG_INFERENCE]
            batch_emb = torch.stack([emb_dict[i] for i in batch_indices])
            if USE_TAG_PROJECTOR:
                # Probabilities first (hidden not needed), then projector
                logits_or_probs = tag_model(batch_emb)  # already probs if sigmoid output
                tag_vec = tag_projector_model(logits_or_probs)
            else:
                logits_or_probs, hidden = tag_model(batch_emb, return_hidden=True)
                tag_vec = hidden  # use hidden representation
            for bi, v in zip(batch_indices, tag_vec):
                vectors[bi] = v
    return vectors

# Global references set during main
synergy_model = None
tag_model = None
tag_projector_model = None


def main():
    global synergy_model, tag_model, tag_projector_model
    debug = False

    # Sanity check synergy architecture vs representation choice
    if USE_TAG_PROJECTOR and "TagHidden" in SYNERGY_ARCH:
        raise ValueError("SYNERGY_ARCH expects hidden states but USE_TAG_PROJECTOR=True")
    if (not USE_TAG_PROJECTOR) and ("TagHidden" not in SYNERGY_ARCH):
        raise ValueError("SYNERGY_ARCH expects projector vectors but USE_TAG_PROJECTOR=False")

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

    print("📦 Loading models (synergy + tag[ + projector]) ...")
    synergy_model, tag_model, tag_projector_model = load_models()

    print("⚙️ Extracting embeddings from cards...")
    emb_dict = extract_embeddings(all_cards, idx_to_name)

    print("🧪 Computing tag representations (this may take a moment)...")
    tag_vec_dict = compute_tag_representations(emb_dict)

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

    autocast_ctx = (
        torch.amp.autocast("cuda")
        if (MIXED_PRECISION and DEVICE.type == "cuda")
        else torch.cpu.amp.autocast(enabled=False)
    )

    for batch in tqdm(
        batch_pairs(n, CHUNK_SIZE, start_idx=last_idx),
        desc="Processing pairs",
        total=remaining_batches,
    ):
        # Build tensors
        emb_a_batch = torch.stack([emb_dict[i] for i, j in batch])
        emb_b_batch = torch.stack([emb_dict[j] for i, j in batch])
        tag_a_batch = torch.stack([tag_vec_dict[i] for i, j in batch])
        tag_b_batch = torch.stack([tag_vec_dict[j] for i, j in batch])

        with torch.no_grad():
            with autocast_ctx:
                logits = synergy_model(emb_a_batch, emb_b_batch, tag_a_batch, tag_b_batch)
                scores = torch.sigmoid(logits).squeeze().float().cpu().tolist()

        for (i, j), score in zip(batch, scores):
            cur.execute(
                "INSERT OR REPLACE INTO synergies (idx_a, idx_b, score) VALUES (?, ?, ?)",
                (i, j, score),
            )
        conn.commit()

        # Update progress based on first i in batch
        current_i = batch[0][0]
        set_last_processed_index(conn, current_i)

        done_pairs += len(batch)
        session_pairs += len(batch)

        if session_pairs % 10_000_000 < len(batch):
            print("\n------ Progress Update ------")
            remaining_pairs = total_possible_pairs - done_pairs
            print(f"\n📈 Progress Update:")
            print(f"   ✅ Total processed: {done_pairs:,}")
            print(f"   🔄 Session processed: {session_pairs:,}")
            print(f"   ⏳ Remaining: {remaining_pairs:,}")
            print(f"   📊 Total: {total_possible_pairs:,}\n")

    conn.close()
    remaining_pairs = total_possible_pairs - done_pairs
    print("\n------------------------------------")
    print("✅ All pairs processed and saved to database.")
    print(f"\n📈 Progress Update:")
    print(f"   ✅ Total processed: {done_pairs:,}")
    print(f"   🔄 Session processed: {session_pairs:,}")
    print(f"   ⏳ Remaining: {remaining_pairs:,}")
    print(f"   📊 Total: {total_possible_pairs:,}\n")


if __name__ == "__main__":
    main()
