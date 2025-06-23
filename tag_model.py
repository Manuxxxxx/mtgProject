import torch
import torch.nn as nn
import torch.nn.functional as F

class TagModel(nn.Module):
    def __init__(
        self,
        input_dim=384,
        hidden_dims=[512, 256],
        output_dim=271,
        dropout=0.3,
        use_batchnorm=True,
        use_sigmoid_output=False
    ):
        """
        Improved tag prediction model with better regularization and flexibility.
        
        Args:
            input_dim: Input embedding size (e.g. from BERT)
            hidden_dims: Tuple of hidden layer sizes
            output_dim: Number of tag classes
            dropout: Dropout rate
            use_batchnorm: Whether to apply batch norm
            use_sigmoid_output: Whether to apply sigmoid in the output layer (for inference)
        """
        super().__init__()

        self.use_sigmoid_output = use_sigmoid_output
        self.use_batchnorm = use_batchnorm

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0]) if use_batchnorm else nn.Identity()
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1]) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.output(x)
        if self.use_sigmoid_output:
            return torch.sigmoid(x)
        return x
    
def init_tag_model_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def build_tag_model(
    arch_name: str,
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    dropout: float = 0.3,
    use_batchnorm: bool = True,
    use_sigmoid_output: bool = False
):
    if arch_name != "tagModel":
        raise ValueError(f"Unknown tag model architecture: {arch_name}")
    
    model = TagModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        use_sigmoid_output=use_sigmoid_output
    )
    model.apply(init_tag_model_weights)
    return model
