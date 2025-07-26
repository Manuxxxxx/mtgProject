import os
import json
import re
import torch
import time
from tqdm import tqdm
import mtgProject.src.training_utils.bert_parsing as bert_parsing
from mtgProject.src.models.bert_model import build_bert_model
from mtgProject.src.models.tag_model import build_tag_model
from mtgProject.src.models.tag_projector_model import build_tag_projector_model


BERT_MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_DIM = 384  
MAX_LEN = 280
BERT_CHECKPOINT_FILE="checkpoints/two_phase_joint/two_phase_joint_training_tag_20250717_122452/bert_multi_model_epoch_18.pth"

TAG_HIDDEN_DIMS = [512, 256]
TAG_OUTPUT_DIM = 103
TAG_DROPOUT = 0.3
TAG_USE_SIGMOID_OUTPUT = True
TAG_CHECKPOINT_FILE = "checkpoints/two_phase_joint/two_phase_joint_training_tag_20250717_122452/tag_multi_model_epoch_18.pth"

TAG_PROJECTOR_OUTPUT_DIM = 64
TAG_PROJECTOR_HIDDEN_DIM = 32
TAG_PROJECTOR_DROPOUT = 0.3
TAG_PROJECTOR_CHECKPOINT_FILE = "checkpoints/two_phase_joint/two_phase_joint_training_tag_20250717_122452/tag_projector_model_epoch_18.pth"


def get_embedding_from_card(card, model, tokenizer, device):
    if card is not None:
        text = bert_parsing.format_card_for_bert(card)
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            return outputs.cpu().numpy()
    else:
        return None

def get_tags_and_projection_from_emb(emb, tag_model, tag_projector_model, device):
    """
    Given an embedding, calculates the tags and their projection.
    Returns a tuple of (tags, projection).
    """
    emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        tags = tag_model(emb_tensor)
        projection = tag_projector_model(tags)
    
    return tags.cpu().numpy(), projection.cpu().numpy()
    
def minify_large_arrays(json_str):
    """
    Flattens the arrays for "emb", "tags", and "tags_projection" into one line,
    whether they are 2D ([[...]]) or 3D ([[[...]]]) arrays.
    """

    target_fields = ["emb_predicted", "tags_predicted", "tags_preds_projection"]

    for field in target_fields:
        # Match both [[...]] and [[[...]]]
        pattern = re.compile(
            rf'"({field})"\s*:\s*(\[\s*(?:\[\s*)*(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*,?\s*)+(?:\s*\]\s*)*\])',
            re.DOTALL
        )

        def replacer(match):
            key = match.group(1)
            array_text = match.group(2)
            # Remove all internal whitespace/newlines
            compact = re.sub(r'\s+', '', array_text)
            return f'"{key}": {compact}'

        json_str = pattern.sub(replacer, json_str)

    return json_str

def load_bulk_file(bulk_file):
    with open(bulk_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_embedding_file(bulk_file, save_every=2000, calculate_tags=False, output_file=None):
    model, tokenizer, device = build_bert_model(BERT_MODEL_NAME, EMBEDDING_DIM)
    model.load_state_dict(torch.load(BERT_CHECKPOINT_FILE))
    model.eval()

    if calculate_tags:
        print("Calculating tags and tags projector")
        tag_model = build_tag_model(
            arch_name="tagModel",
            input_dim=EMBEDDING_DIM,
            hidden_dims=TAG_HIDDEN_DIMS,
            output_dim=TAG_OUTPUT_DIM,
            dropout=TAG_DROPOUT,
            use_batchnorm=None,
            use_sigmoid_output=TAG_USE_SIGMOID_OUTPUT
        ).to(device)
        tag_model.load_state_dict(torch.load(TAG_CHECKPOINT_FILE))
        tag_model.eval()

        tag_projector_model = build_tag_projector_model(
            num_tags=TAG_OUTPUT_DIM,
            hidden_dim=TAG_PROJECTOR_HIDDEN_DIM,
            output_dim=TAG_PROJECTOR_OUTPUT_DIM,
            dropout=TAG_PROJECTOR_DROPOUT
        ).to(device)
        tag_projector_model.load_state_dict(torch.load(TAG_PROJECTOR_CHECKPOINT_FILE))
        tag_projector_model.eval()

    cards = load_bulk_file(bulk_file)
    counter_save = 0
    for card in tqdm(cards, desc="Processing cards"):
        card_emb = get_embedding_from_card(card, model, tokenizer, device)
        if card_emb is not None:
            card["emb_predicted"] = card_emb.tolist() 
            if calculate_tags:
                tags, projection = get_tags_and_projection_from_emb(card_emb, tag_model, tag_projector_model, device)
                card["tags_predicted"] = tags.tolist()
                card["tags_preds_projection"] = projection.tolist()
        else:
            #error handling
            print(f"Error processing card: {card.get('name', 'Unknown')}")


        counter_save += 1

        if counter_save > save_every:
            
            with open(output_file, "w", encoding="utf-8") as f:
                pretty_json = json.dumps(cards, indent=4)
                compacted = minify_large_arrays(pretty_json)
                f.write(compacted)
            counter_save = 0
                
        
        

    # #save

    with open(output_file, "w", encoding="utf-8") as f:
        pretty_json = json.dumps(cards, indent=2)
        compacted = minify_large_arrays(pretty_json)
        f.write(compacted)

if __name__ == "__main__":
    time = time.strftime("%Y%m%d%H%M%S")
    os.makedirs("datasets/processed/embedding_predicted/joint_tag", exist_ok=True)
    output_file = f"datasets/processed/embedding_predicted/joint_tag/cards_with_tags_{time}.json"
    create_embedding_file("datasets/processed/tag_included/cards_with_tags_103_20250627182008.json", save_every=5000, output_file=output_file, calculate_tags=True)
