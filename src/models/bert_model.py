import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, BertModel,
    DistilBertTokenizer, DistilBertModel,
    AutoModel, AutoTokenizer
)

# ----------------------
# Model
# ----------------------
class EmbedRegressor(nn.Module):
    def __init__(self, output_dim, model_name):
        super().__init__()
        self.model_name = model_name

        if model_name == "bert-base-uncased":
            print("Using BertModel")
            self.encoder = BertModel.from_pretrained("bert-base-uncased")

        elif model_name == "distilbert-base-uncased":
            print("Using DistilBertModel")
            self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        elif model_name == "microsoft/deberta-v3-small":
            print("Using DeBERTa-v3-small")
            self.encoder = AutoModel.from_pretrained("microsoft/deberta-v3-small")

        elif model_name == "roberta-base":
            print("Using RoBERTa-base")
            self.encoder = AutoModel.from_pretrained("roberta-base")

        elif model_name == "sentence-transformers/all-MiniLM-L6-v2":
            print("Using MiniLM (Sentence-Transformers)")
            self.encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        else:
            raise ValueError(f"Model {model_name} is not supported for this task.")

        hidden_size = self.encoder.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

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
    # Freezing helpers
    # ----------------------
    def freeze_encoder_layers(self, freeze_until_layer):
        """
        Freeze encoder layers up to (but not including) freeze_until_layer.
        """
        if self.model_name == "bert-base-uncased":
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < freeze_until_layer:
                    for param in layer.parameters():
                        param.requires_grad = False

        elif self.model_name == "distilbert-base-uncased":
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.encoder.transformer.layer):
                if i < freeze_until_layer:
                    for param in layer.parameters():
                        param.requires_grad = False

        elif self.model_name in ["microsoft/deberta-v3-small", "roberta-base", "sentence-transformers/all-MiniLM-L6-v2"]:
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < freeze_until_layer:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def unfreeze_encoder_layers(self, start_layer=0):
        """
        Unfreeze encoder layers starting from start_layer.
        """
        if self.model_name == "bert-base-uncased":
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i >= start_layer:
                    for param in layer.parameters():
                        param.requires_grad = True

        elif self.model_name == "distilbert-base-uncased":
            for i, layer in enumerate(self.encoder.transformer.layer):
                if i >= start_layer:
                    for param in layer.parameters():
                        param.requires_grad = True

        elif self.model_name in ["microsoft/deberta-v3-small", "roberta-base", "sentence-transformers/all-MiniLM-L6-v2"]:
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i >= start_layer:
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def freeze_all(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


def build_bert_model(model_name, embedding_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    elif model_name == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    else:
        # DeBERTa, RoBERTa, MiniLM, etc.
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = EmbedRegressor(output_dim=embedding_dim, model_name=model_name).to(device)
    return model, tokenizer, device
