import torch.nn as nn

class MultiTaskProjector(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, tag_dim=256, synergy_dim=256, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.tag_head = nn.Linear(hidden_dim, tag_dim)
        self.synergy_head = nn.Linear(hidden_dim, synergy_dim)

    def forward(self, x):
        shared = self.shared(x)
        tag_embedding = self.tag_head(shared)
        synergy_embedding = self.synergy_head(shared)
        return tag_embedding, synergy_embedding

# ----------------------
# Weight Initialization
# ----------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# ----------------------
# Model Factory
# ----------------------
def build_multitask_projector_model(input_dim=384, hidden_dim=256, tag_dim=256, synergy_dim=256, dropout=0.1):
    """
    Build and initialize a MultiTaskProjector model.

    Args:
        input_dim (int): Input embedding dimension from BERT.
        hidden_dim (int): Shared hidden layer size.
        tag_dim (int): Output dimension for tag-specific head.
        synergy_dim (int): Output dimension for synergy-specific head.
        dropout (float): Dropout rate.

    Returns:
        nn.Module: Initialized MultiTaskProjector instance.
    """
    projector = MultiTaskProjector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        tag_dim=tag_dim,
        synergy_dim=synergy_dim,
        dropout=dropout
    )
    projector.apply(init_weights)
    return projector