import torch
import torch.nn as nn
from itertools import chain
from transformers import (
    BertTokenizer, BertModel,
    DistilBertTokenizer, DistilBertModel,
    AutoModel, AutoTokenizer
)

# ----------------------
# Model
# ----------------------
class BertEmbedRegressor(nn.Module):
    def __init__(self, output_dim, model_name):
        super().__init__()
        self.model_name = model_name

        if model_name == "bert-base-uncased":
            print("Using BertModel")
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        elif model_name == "distilbert-base-uncased":
            print("Using DistilBertModel")
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        elif model_name == "microsoft/deberta-v3-small":
            print("Using DeBERTa-v3-small")
            self.bert = AutoModel.from_pretrained("microsoft/deberta-v3-small")

        elif model_name == "roberta-base":
            print("Using RoBERTa-base")
            self.bert = AutoModel.from_pretrained("roberta-base")

        elif model_name == "sentence-transformers/all-MiniLM-L6-v2":
            print("Using MiniLM (Sentence-Transformers)")
            self.bert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        else:
            raise ValueError(f"Model {model_name} is not supported for this task.")

        hidden_size = self.bert.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Different models expose outputs differently
        if self.model_name == "bert-base-uncased":
            cls_emb = outputs.pooler_output  # already pooled [CLS]

        elif self.model_name == "distilbert-base-uncased":
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        elif self.model_name in ["microsoft/deberta-v3-small", "roberta-base", "sentence-transformers/all-MiniLM-L6-v2"]:
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return self.linear(self.dropout(self.layer_norm(cls_emb)))

    # ----------------------
    # Optimizer param group helpers (backbone vs head)
    # ----------------------
    def backbone_parameters(self):
        """Iterator of backbone (transformer) params."""
        return self.bert.parameters()

    def head_parameters(self):
        """Iterator of head params (norm + projection)."""
        # dropout has no params
        return chain(self.layer_norm.parameters(), self.linear.parameters())

    def get_parameter_groups(self, lr_backbone, lr_head, weight_decay=0.0):
        """
        Convenience helper to build optimizer param groups with different LRs.
        """
        return [
            {"params": self.backbone_parameters(), "lr": lr_backbone, "weight_decay": weight_decay, "name": "bert_backbone"},
            {"params": self.head_parameters(), "lr": lr_head, "weight_decay": weight_decay, "name": "bert_head"},
        ]

    # ----------------------
    # Freezing helpers
    # ----------------------
    def _get_transformer_layers(self):
        """
        Return the ModuleList of transformer layers for the current HF model.
        Works for BERT/DeBERTa/RoBERTa/MiniLM/DistilBERT.
        """
        # BERT / RoBERTa / DeBERTa / MiniLM variants
        if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            return self.bert.encoder.layer
        # DistilBERT
        if hasattr(self.bert, "transformer") and hasattr(self.bert.transformer, "layer"):
            return self.bert.transformer.layer
        raise ValueError(f"Unsupported transformer structure for model: {self.model_name}")

    def _freeze_embeddings(self):
        emb = getattr(self.bert, "embeddings", None)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad = False

    def freeze_encoder_layers(self, freeze_until_layer):
        """
        Freeze embeddings and encoder layers up to (but not including) freeze_until_layer.
        """
        # Always freeze embeddings for stability
        self._freeze_embeddings()

        layers = self._get_transformer_layers()
        for i, layer in enumerate(layers):
            if i < freeze_until_layer:
                for p in layer.parameters():
                    p.requires_grad = False

    def unfreeze_encoder_layers(self, start_layer=0, unfreeze_embeddings=False):
        """
        Unfreeze encoder layers starting from start_layer.
        Set unfreeze_embeddings=True to also unfreeze embeddings.
        """
        if unfreeze_embeddings:
            emb = getattr(self.bert, "embeddings", None)
            if emb is not None:
                for p in emb.parameters():
                    p.requires_grad = True

        layers = self._get_transformer_layers()
        for i, layer in enumerate(layers):
            if i >= start_layer:
                for p in layer.parameters():
                    p.requires_grad = True

    def num_encoder_layers(self):
        return len(self._get_transformer_layers())

    def freeze_all(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.bert.parameters():
            param.requires_grad = True


def build_bert_model(model_name, embedding_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    elif model_name == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    else:
        # DeBERTa, RoBERTa, MiniLM, etc.
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = BertEmbedRegressor(output_dim=embedding_dim, model_name=model_name).to(device)
    return model, tokenizer, device
