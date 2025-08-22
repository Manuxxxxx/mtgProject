import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3, use_batchnorm=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_batchnorm else None
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with optional projection when in_dim != out_dim.
    Layout: BN -> ReLU -> FC -> BN -> ReLU -> Dropout -> FC (+ skip) -> ReLU
    """
    def __init__(self, in_dim, out_dim, dropout=0.3, use_batchnorm=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.bn1 = nn.BatchNorm1d(in_dim) if use_batchnorm else nn.Identity()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.proj(x)
        out = self.bn1(x)
        out = self.act(out)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = out + identity
        out = self.act(out)
        return out


class SelfAttentionBlock(nn.Module):
    """
    Channel-attention (SE-style) block suitable for single-vector features.
    gate = sigmoid(W2(ReLU(W1(x)))) ; y = x * gate ; then FFN to out_dim.
    """
    def __init__(self, in_dim, out_dim, dropout=0.3, reduction=4):
        super().__init__()
        hidden = max(in_dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a = self.gate(x)
        x = x * a
        x = self.ffn(x)
        return x


class TagModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        use_sigmoid_output: bool = False,
        model_type: str = "simple",  # "simple" | "residual" | "attention"
    ):
        super().__init__()
        self.use_sigmoid_output = use_sigmoid_output
        self.model_type = model_type

        dims = [input_dim] + (hidden_dims or [])
        blocks = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            if model_type == "simple":
                blocks.append(MLPBlock(in_d, out_d, dropout=dropout, use_batchnorm=use_batchnorm))
            elif model_type == "residual":
                blocks.append(ResidualBlock(in_d, out_d, dropout=dropout, use_batchnorm=use_batchnorm))
            elif model_type == "attention":
                blocks.append(SelfAttentionBlock(in_d, out_d, dropout=dropout))
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        self.blocks = nn.ModuleList(blocks)
        last_hidden = dims[-1] if hidden_dims else input_dim
        self.output_layer = nn.Linear(last_hidden, output_dim)

    def forward(self, x, return_hidden: bool = False):
        h = x
        for blk in self.blocks:
            h = blk(h)
        logits = self.output_layer(h)
        if self.use_sigmoid_output:
            logits = torch.sigmoid(logits)
        if return_hidden:
            return logits, h
        return logits


def init_tag_model_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, (nn.BatchNorm1d,)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def build_tag_model(
    arch_name: str,
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    dropout: float = 0.3,
    use_batchnorm: bool = True,
    use_sigmoid_output: bool = False,
):
    """
    arch_name options (kept compatible):
      - "tagModel" or "simple" -> simple MLP
      - "residual" or "tagModelResidual" -> residual MLP
      - "attention" or "tagModelAttention" -> channel-attention blocks
    """

    model_type = {
        "tagModel": "simple",
        "simple": "simple",
        "residual": "residual",
        "tagModelResidual": "residual",
        "attention": "attention",
        "tagModelAttention": "attention",
    }.get(arch_name, "simple")

    model = TagModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        use_sigmoid_output=use_sigmoid_output,
        model_type=model_type,
    )
    # Optional: apply init
    model.apply(init_tag_model_weights)
    return model


def load_legacy_tag_weights(new_model, ckpt_path, device):
    """
    Load checkpoints saved with old TagModel that used layer names:
      fc1.weight / fc1.bias
      fc2.weight / fc2.bias
      (optionally fc3...) 
      output.weight / output.bias
    Map them to new:
      blocks.{i}.fc.weight / blocks.{i}.fc.bias
      output_layer.weight / output_layer.bias

    Falls back silently (logs) if shapes differ or layers count changed.
    """
    sd = torch.load(ckpt_path, map_location=device)
    # Allow either direct state_dict or wrapped
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    old_sd = sd

    new_sd = new_model.state_dict()

    # Collect old hidden layer names in order (fc1, fc2, ...)
    old_hidden_layers = []
    for k in old_sd.keys():
        if k.startswith("fc") and (k.endswith(".weight") or k.endswith(".bias")):
            # extract index after 'fc'
            try:
                idx = int(k[2:k.index(".")])
                old_hidden_layers.append((idx, k))
            except ValueError:
                pass
    # Group by layer index
    max_idx = 0
    layer_params = {}
    for idx, k in old_hidden_layers:
        layer_params.setdefault(idx, {})
        if k.endswith(".weight"):
            layer_params[idx]["weight"] = old_sd[k]
        else:
            layer_params[idx]["bias"] = old_sd[k]
        max_idx = max(max_idx, idx)

    # Sort by index (fc1, fc2, ...)
    ordered_layers = [layer_params[i] for i in sorted(layer_params.keys())]

    # Map each old fc{i} to blocks.{i-1}.fc
    applied = []
    for i, layer in enumerate(ordered_layers):
        target_w = f"blocks.{i}.fc.weight"
        target_b = f"blocks.{i}.fc.bias"
        if "weight" in layer and target_w in new_sd and new_sd[target_w].shape == layer["weight"].shape:
            new_sd[target_w] = layer["weight"]
            applied.append(f"{target_w} <- fc{i+1}.weight")
        if "bias" in layer and target_b in new_sd and new_sd[target_b].shape == layer["bias"].shape:
            new_sd[target_b] = layer["bias"]
            applied.append(f"{target_b} <- fc{i+1}.bias")

    # Output layer mapping
    if "output.weight" in old_sd and "output_layer.weight" in new_sd and new_sd["output_layer.weight"].shape == old_sd["output.weight"].shape:
        new_sd["output_layer.weight"] = old_sd["output.weight"]
        applied.append("output_layer.weight <- output.weight")
    if "output.bias" in old_sd and "output_layer.bias" in new_sd and new_sd["output_layer.bias"].shape == old_sd["output.bias"].shape:
        new_sd["output_layer.bias"] = old_sd["output.bias"]
        applied.append("output_layer.bias <- output.bias")

    missing_before, unexpected_before = new_model.load_state_dict(new_sd, strict=False)

    print("Legacy tag weight remap applied:")
    for line in applied:
        print("  ", line)
    if missing_before:
        print("Remaining missing keys:", missing_before)
    if unexpected_before:
        print("Unexpected keys (ignored):", unexpected_before)

    return new_model
