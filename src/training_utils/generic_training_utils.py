import sys
import random
import numpy as np
import json
from src.training_utils import bert_parsing

ANSI_COLORS = {
    "reset": "\033[0m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bold": "\033[1m",
}

# Store original stdout
original_write = sys.stdout.write

# Current color state
current_color = ""


def set_color(color_name=None):
    """
    Set the global color for all print() calls.
    Call with None or 'reset' to reset to default.
    """
    global current_color
    if color_name is None or color_name == "reset":
        current_color = ""
        sys.stdout.write = original_write  # Restore default
    else:
        color_code = ANSI_COLORS.get(color_name.lower())
        if not color_code:
            raise ValueError(f"Unknown color: {color_name}")

        current_color = color_code

        def color_write(text):
            original_write(
                color_code + text + ANSI_COLORS["reset"] if text.strip() else text
            )

        sys.stdout.write = color_write


def calculate_stats_lenght_tokenizer_and_plot(tokenizer, cards, max_length=450):
    """
    Calculate the average and max length of tokenized cards using the provided tokenizer.
    Plots all the lengths in a graph.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        cards (list): List of card dictionaries to tokenize.
        max_length (int): Maximum length for tokenization.

    Returns:
        None: This function does not return anything, but it will plot a histogram of token lengths.
    """
    # plot histogram of lengths and save it
    import matplotlib.pyplot as plt

    tokenized_lengths = [
        len(
            tokenizer(
                bert_parsing.format_card_for_bert(card),
                truncation=True,
                max_length=max_length,
            )["input_ids"]
        )
        for card in cards
    ]
    avg_length = np.mean(tokenized_lengths)
    max_length = np.max(tokenized_lengths)
    min_length = np.min(tokenized_lengths)
    plt.hist(tokenized_lengths, bins=50, alpha=0.75)
    plt.title("Tokenized Lengths Histogram")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.axvline(
        avg_length,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Avg Length: {avg_length:.2f}",
    )
    plt.axvline(
        max_length,
        color="g",
        linestyle="dashed",
        linewidth=1,
        label=f"Max Length: {max_length}",
    )
    plt.axvline(
        min_length,
        color="b",
        linestyle="dashed",
        linewidth=1,
        label=f"Min Length: {min_length}",
    )
    plt.grid(True)
    plt.savefig("tokenized_lengths_histogram.png")
    plt.close()


def set_seed(seed: int = 42):
    random.seed(seed)  # Python random module
    # np.random.seed(seed)                   # NumPy
    # torch.manual_seed(seed)                # PyTorch CPU
    # torch.cuda.manual_seed(seed)           # PyTorch GPU
    # torch.cuda.manual_seed_all(seed)       # All GPUs (if using DataParallel or DDP)

    # # Ensures deterministic behavior (at the expense of performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_real_fake_indices(synergy_file):
    """
    Load the synergy file and return the indices for real and fake entries.
    Real = contains "synergy_edhrec"
    Fake = does not contain it and synergy == 0
    """
    with open(synergy_file, "r") as f:
        data = json.load(f)

    real_indices = []
    fake_indices = []

    for i, entry in enumerate(data):
        if "synergy_edhrec" in entry:
            real_indices.append(i)
        elif entry.get("synergy", 0) == 0:
            fake_indices.append(i)

    return real_indices, fake_indices


def print_separator():
    print("=" * 30)


def print_models_param_summary(models_with_names, optimizer):
    """
    Args:
        models_with_names: list of tuples (model_name, model_instance)
        optimizer: the optimizer containing param_groups
    """
    print("Optimizer name:", optimizer.__class__.__name__)
    print("Optimizer parameter groups:")
    for param_group in optimizer.param_groups:
        print(
            f"  - Learning rate: {param_group['lr']}, "
            f"Params: {len(param_group['params'])}, "
            f"Name: {param_group.get('name', 'N/A')}"
        )

    print("\nModel parameter summary:")
    for name, model in models_with_names:
        if model is None:
            print(f"Model '{name}': None")
            continue

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"Model '{name}':")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Frozen parameters: {frozen_params:,}")


def split_indices(real_indices, fake_indices, splits, log_splits=False):
    random.shuffle(real_indices)
    random.shuffle(fake_indices)

    num_real = len(real_indices)
    num_fake = len(fake_indices)

    real_ptr = 0
    fake_ptr = 0

    real_allocations = {}
    fake_allocations = {}

    # Allocate real indices
    for split_name, ratios in splits.items():
        count = int(ratios.get("real", 0) * num_real)
        real_allocations[split_name] = real_indices[real_ptr : real_ptr + count]
        real_ptr += count

    # Allocate fake indices
    for split_name, ratios in splits.items():
        count = int(ratios.get("fake", 0) * num_fake)
        fake_allocations[split_name] = fake_indices[fake_ptr : fake_ptr + count]
        fake_ptr += count

    # Combine real and fake indices per split
    final_splits = {}
    for split_name in splits:
        final_splits[split_name] = real_allocations.get(
            split_name, []
        ) + fake_allocations.get(split_name, [])
        if log_splits:
            print(
                f"{split_name} - Real: {len(real_allocations[split_name])}, Fake: {len(fake_allocations[split_name])}, Total: {len(final_splits[split_name])}"
            )

    return final_splits


def get_loss_tag_fn(config, device, tag_model_pos_weight=None):
    """
    Build the loss function for the tag model based on the configuration.
    If use_focal is True, use FocalLoss; otherwise, use weighted BCE.
    """
    if config.get("use_focal", False):
        # Use FocalLoss, normalize alpha if provided
        if tag_model_pos_weight is not None:
            alpha = (
                tag_model_pos_weight / tag_model_pos_weight.max()
            )  # Normalize to 0–1
        else:
            alpha = None

        loss_tag_fn = FocalLoss(alpha=alpha, gamma=config.get("focal_gamma", 2.0)).to(
            device
        )

        print(
            "Using Focal Loss for tag model with alpha:",
            alpha,
            "and gamma:",
            config.get("focal_gamma", 2.0),
        )

    else:
        # Use weighted BCE
        loss_tag_fn = nn.BCEWithLogitsLoss(pos_weight=tag_model_pos_weight).to(device)
        print(
            "Using weighted BCE Loss for tag model with pos_weight:",
            tag_model_pos_weight,
        )

    return loss_tag_fn
