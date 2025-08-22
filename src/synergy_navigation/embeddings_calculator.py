import os
import json
import re
import torch
import time
from tqdm import tqdm
import numpy as np
import src.training_utils.bert_parsing as bert_parsing
from src.models.bert_model import build_bert_model
from src.models.tag_model import build_tag_model, remap_legacy_tag_state_dict
from src.models.tag_projector_model import build_tag_projector_model
from src.models.multitask_projector_model import build_multitask_projector_model


BERT_MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_DIM = 384
MAX_LEN = 256
BERT_CHECKPOINT_FILE = "checkpoints/two_phase_joint/two_phase_joint__tag_hidden__detach_20250822_150413/bert_multi_model_epoch_44.pth"

# Tag model config (defaults; can be overridden via function args)
TAG_HIDDEN_DIMS = [512, 512]
TAG_OUTPUT_DIM = 174
TAG_DROPOUT = 0.3
TAG_USE_SIGMOID_OUTPUT = True  # At inference you might want probabilities
TAG_ARCH = "simple"
TAG_CHECKPOINT_FILE = "checkpoints/two_phase_joint/two_phase_joint__tag_hidden__detach_20250822_150413/tag_multi_model_epoch_44.pth"

# Projector defaults (optional)
TAG_PROJECTOR_CHECKPOINT_FILE = None  # e.g. "checkpoints/.../tag_projector_model_epoch_XX.pth"
MULTITASK_PROJECTOR_CHECKPOINT_FILE = None
MULTITASK_PROJECTOR_TAG_DIM = None
MULTITASK_PROJECTOR_SYNERGY_DIM = None
MULTITASK_PROJECTOR_HIDDEN_DIM = None
MULTITASK_PROJECTOR_DROPOUT = 0.3


def _format_cards_for_bert(cards):
    texts = []
    for card in cards:
        texts.append(bert_parsing.format_card_for_bert(card))
    return texts


def batch_encode(texts, tokenizer, max_len):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )


def get_embeddings_from_cards(cards, model, tokenizer, device, batch_size):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(cards), batch_size), desc="Embedding cards", unit="batch"):
            batch_cards = cards[i:i + batch_size]
            texts = _format_cards_for_bert(batch_cards)
            enc = batch_encode(texts, tokenizer, MAX_LEN)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)  # [B, emb_dim]
            # Directly convert to list-of-lists (JSON friendly) to avoid numpy arrays
            all_embs.extend(outputs.cpu().tolist())
    return all_embs


def _to_python(obj):
    """Recursively convert tensors / numpy objects to pure Python types."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, list):
        return [_to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    return obj


def run_tag_inference(emb_batch_tensor, tag_model, use_sigmoid_output):
    with torch.no_grad():
        logits_or_probs = tag_model(emb_batch_tensor)
        if not use_sigmoid_output:  # logits -> probabilities for storage
            probs = torch.sigmoid(logits_or_probs)
        else:
            probs = logits_or_probs
    return probs


def apply_tag_projector(tag_probs, tag_projector_model):
    with torch.no_grad():
        proj = tag_projector_model(tag_probs)
    return proj


def apply_multitask_projector(emb_batch_tensor, multitask_projector_model):
    with torch.no_grad():
        tag_emb, synergy_emb = multitask_projector_model(emb_batch_tensor)
    return tag_emb, synergy_emb


def minify_large_arrays(json_str):
    target_fields = ["emb_predicted", "tags_predicted", "tags_preds_projection"]
    for field in target_fields:
        pattern = re.compile(
            rf'"({field})"\s*:\s*(\[\s*(?:\[\s*)*(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*,?\s*)+(?:\s*\]\s*)*\])',
            re.DOTALL,
        )

        def replacer(match):
            key = match.group(1)
            array_text = match.group(2)
            compact = re.sub(r"\s+", "", array_text)
            return f'"{key}": {compact}'

        json_str = pattern.sub(replacer, json_str)
    return json_str


def load_bulk_file(bulk_file):
    with open(bulk_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def safe_load_tag_model(tag_model, checkpoint_path, device, force_no_batchnorm=False):
    """
    Attempt to load a tag model checkpoint with backward compatibility.
    1. Try strict load.
    2. If fails due to missing bn.* keys, optionally rebuild without BN or load with strict=False.
    3. If legacy (fc1 / output) naming detected, remap and load strict=False.
    """
    sd = torch.load(checkpoint_path, map_location=device)
    # Unwrap possible wrappers
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    try:
        tag_model.load_state_dict(sd, strict=True)
        print(f"Loaded tag model (strict) from {checkpoint_path}")
        return tag_model
    except RuntimeError as e:
        msg = str(e)
        print(f"Strict load failed: {msg}")

    # Detect legacy naming
    legacy = any(k.startswith("fc1") for k in sd.keys()) or "output.weight" in sd
    missing_bn = any("bn.running_mean" in line for line in msg.splitlines())

    if legacy:
        print("Detected legacy tag checkpoint layout -> remapping.")
        remapped = remap_legacy_tag_state_dict(sd, tag_model)
        missing, unexpected = tag_model.load_state_dict(remapped, strict=False)
        print("Remap load done. Missing:", missing, "Unexpected:", unexpected)
        return tag_model

    if missing_bn and force_no_batchnorm:
        print("Rebuilding TagModel without BatchNorm to match checkpoint (missing bn parameters).")
        # Rebuild without BN using same dims
        rebuilt = build_tag_model(
            arch_name="simple",  # assumes simple; adjust if you pass different arch
            input_dim=tag_model.blocks[0].fc.in_features if tag_model.blocks else tag_model.output_layer.in_features,
            hidden_dims=[blk.fc.out_features for blk in tag_model.blocks] if hasattr(tag_model.blocks[0], "fc") else [],
            output_dim=tag_model.output_layer.out_features,
            dropout=0.0,
            use_batchnorm=False,
            use_sigmoid_output=tag_model.use_sigmoid_output,
        ).to(device)
        missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
        print("Loaded without BN. Missing:", missing, "Unexpected:", unexpected)
        return rebuilt

    # Last fallback: non-strict load
    print("Falling back to non-strict load (weights with matching names loaded).")
    missing, unexpected = tag_model.load_state_dict(sd, strict=False)
    print("Missing:", missing, "Unexpected:", unexpected)
    return tag_model


def create_embedding_file(
    bulk_file,
    output_file,
    calculate_tags=False,
    batch_size=64,
    tag_hidden_dims=None,
    tag_output_dim=TAG_OUTPUT_DIM,
    tag_dropout=TAG_DROPOUT,
    tag_use_sigmoid_output=TAG_USE_SIGMOID_OUTPUT,
    tag_arch=TAG_ARCH,
    tag_checkpoint_file=TAG_CHECKPOINT_FILE,
    use_tag_projector=False,
    tag_projector_checkpoint=TAG_PROJECTOR_CHECKPOINT_FILE,
    tag_projector_output_dim=None,
    tag_projector_hidden_dim=None,
    tag_projector_dropout=0.3,
    use_multitask_projector=False,
    multitask_projector_checkpoint=MULTITASK_PROJECTOR_CHECKPOINT_FILE,
    multitask_projector_tag_dim=MULTITASK_PROJECTOR_TAG_DIM,
    multitask_projector_synergy_dim=MULTITASK_PROJECTOR_SYNERGY_DIM,
    multitask_projector_hidden_dim=MULTITASK_PROJECTOR_HIDDEN_DIM,
    multitask_projector_dropout=MULTITASK_PROJECTOR_DROPOUT,
):
    """Create an embedding (and optional tag / projection) annotated JSON.

    Overwrites existing tag-related keys if present.
    """
    if tag_hidden_dims is None:
        tag_hidden_dims = TAG_HIDDEN_DIMS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model, tokenizer, _ = build_bert_model(BERT_MODEL_NAME, EMBEDDING_DIM)
    bert_model.load_state_dict(torch.load(BERT_CHECKPOINT_FILE, map_location=device))
    bert_model.to(device).eval()

    multitask_projector_model = None
    tag_projector_model = None

    tag_model_input_dim = EMBEDDING_DIM
    if use_multitask_projector:
        if multitask_projector_tag_dim is None or multitask_projector_synergy_dim is None:
            raise ValueError("Must supply multitask_projector_tag_dim & synergy_dim when use_multitask_projector=True")
        multitask_projector_model = build_multitask_projector_model(
            input_dim=EMBEDDING_DIM,
            tag_dim=multitask_projector_tag_dim,
            synergy_dim=multitask_projector_synergy_dim,
            hidden_dim=multitask_projector_hidden_dim,
            dropout=multitask_projector_dropout,
        ).to(device)
        if multitask_projector_checkpoint:
            multitask_projector_model.load_state_dict(torch.load(multitask_projector_checkpoint, map_location=device))
        multitask_projector_model.eval()
        tag_model_input_dim = multitask_projector_tag_dim

    if calculate_tags:
        tag_model = build_tag_model(
            arch_name=tag_arch,
            input_dim=tag_model_input_dim,
            hidden_dims=tag_hidden_dims,
            output_dim=tag_output_dim,
            dropout=tag_dropout,
            use_batchnorm=False,
            use_sigmoid_output=tag_use_sigmoid_output,
        ).to(device)
        if tag_checkpoint_file:
            tag_model = safe_load_tag_model(tag_model, tag_checkpoint_file, device)

        if use_tag_projector:
            if tag_projector_output_dim is None:
                raise ValueError("tag_projector_output_dim required when use_tag_projector=True")
            tag_projector_model = build_tag_projector_model(
                num_tags=tag_output_dim,
                hidden_dim=tag_projector_hidden_dim,
                output_dim=tag_projector_output_dim,
                dropout=tag_projector_dropout,
            ).to(device)
            if tag_projector_checkpoint:
                tag_projector_model.load_state_dict(torch.load(tag_projector_checkpoint, map_location=device))
            tag_projector_model.eval()

    cards = load_bulk_file(bulk_file)

    # Batch embedding extraction
    embeddings = get_embeddings_from_cards(cards, bert_model, tokenizer, device, batch_size=batch_size)

    # Attach embeddings (and tags) back to cards
    if calculate_tags:
        # Prepare embeddings tensor (optionally project first)
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
        if use_multitask_projector and multitask_projector_model is not None:
            tag_emb_tensor, _ = apply_multitask_projector(emb_tensor, multitask_projector_model)
        else:
            tag_emb_tensor = emb_tensor
        # Tag probabilities
        tag_probs = []
        with torch.no_grad():
            for i in range(0, tag_emb_tensor.size(0), batch_size):
                batch_emb = tag_emb_tensor[i:i + batch_size]
                probs = run_tag_inference(batch_emb, tag_model, tag_use_sigmoid_output)
                tag_probs.append(probs.cpu())
        tag_probs = torch.cat(tag_probs, dim=0)
        if use_tag_projector and tag_projector_model is not None:
            tag_proj = []
            with torch.no_grad():
                for i in range(0, tag_probs.size(0), batch_size):
                    proj = apply_tag_projector(tag_probs[i:i + batch_size].to(device), tag_projector_model)
                    tag_proj.append(proj.cpu())
            tag_proj = torch.cat(tag_proj, dim=0).numpy()
        else:
            tag_proj = None

    # Assign to cards
    for idx, card in tqdm(enumerate(cards), total=len(cards), desc="Annotating cards"):
        # embeddings[idx] is already a list[float]; wrap to keep prior 2D shape convention
        card["emb_predicted"] = [embeddings[idx]]
        if calculate_tags:
            card["tags_predicted"] = [_to_python(tag_probs[idx])]
            if use_tag_projector and tag_proj is not None:
                card["tags_preds_projection"] = [_to_python(tag_proj[idx])]
            elif "tags_preds_projection" in card:
                del card["tags_preds_projection"]

    # Ensure full structure is python-native
    cards = _to_python(cards)

    # Save once at end
    with open(output_file, "w", encoding="utf-8") as f:
        pretty_json = json.dumps(cards, indent=2)
        compacted = minify_large_arrays(pretty_json)
        f.write(compacted)
    print(f"Saved {len(cards)} cards with embeddings to {output_file}")


if __name__ == "__main__":
    ts = time.strftime("%Y%m%d%H%M%S")
    os.makedirs("datasets/processed/embedding_predicted/joint_tag", exist_ok=True)
    N_TAGS = TAG_OUTPUT_DIM
    output_file = (
        f"datasets/processed/embedding_predicted/joint_tag/cards_with_tags_{N_TAGS}_{ts}.json"
    )
    create_embedding_file(
        bulk_file="datasets/processed/tag_included/cards_with_tags_174_20250820145339.json",
        output_file=output_file,
        calculate_tags=True,
        batch_size=128,
        use_tag_projector=False,
        use_multitask_projector=False,
    )
