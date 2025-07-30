import torch
import torch.nn as nn

class TagProjector(nn.Module):
    def __init__(self, num_tags: int, output_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        """
        Compress tag probabilities/logits into a dense embedding for synergy modeling.
        
        Args:
            num_tags (int): Number of tags (input size).
            output_dim (int): Output embedding size.
            hidden_dim (int): Hidden layer size.
            dropout (float): Dropout probability.
        """
        super(TagProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_tags, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, tag_input):
        """
        Args:
            tag_input (Tensor): (batch_size, num_tags) — sigmoid(tag_logits) or tag_logits.
        
        Returns:
            Tensor: (batch_size, output_dim) — projected embedding
        """
        return self.net(tag_input)
    
def build_tag_projector_model(num_tags, output_dim=64, hidden_dim=128, dropout=0.1):
    """
    Build and initialize a TagProjector model.

    Args:
        num_tags (int): Number of tag inputs.
        output_dim (int): Size of the projected embedding.
        hidden_dim (int): Hidden layer size.
        dropout (float): Dropout rate.

    Returns:
        nn.Module: Initialized TagProjector instance.
    """
    projector = TagProjector(
        num_tags=num_tags,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    projector.apply(init_weights)
    return projector

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

