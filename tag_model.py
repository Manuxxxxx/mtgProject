import torch
import torch.nn as nn
import torch.nn.functional as F

class TagModel(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, output_dim=271, dropout=0.2):
        """
        input_dim: dimension of BERT embeddings (usually 768 or 1024)
        hidden_dim: size of intermediate layer
        output_dim: number of tags (e.g., 271 for top-40-min-count tags)
        dropout: dropout rate between layers
        """
        super(TagModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, input_dim)
        returns: logits of shape (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No sigmoid here, handled by BCEWithLogitsLoss
        return x