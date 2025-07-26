import torch.nn as nn
import torch.optim as optim

from mtgProject.src.training_utils.generic_training_utils import print_separator, print_models_param_summary, get_loss_tag_fn


def build_training_components_tag(config, bert_model, tag_model, device, tag_model_pos_weight=None):
    optimizer = build_tag_optimizer(
        optimizer_name=config["optimizer"],
        tag_model=tag_model,
        bert_model=bert_model,
        tag_lr=config["tag_learning_rate_tag"],
        bert_lr=config["bert_learning_rate_tag"],
        optimizer_config=config.get("optimizer_config", {}),
    )

    
    models_with_names = [
        ("bert_model", bert_model),
        ("tag_model", tag_model)
    ]

    print_models_param_summary(models_with_names, optimizer)

    print_separator()

    loss_tag_fn = get_loss_tag_fn(
        config=config,
        device=device,
        tag_model_pos_weight=tag_model_pos_weight
    )

    return optimizer, loss_tag_fn
    
def build_tag_optimizer(
    optimizer_name, tag_model, bert_model, tag_lr, bert_lr, optimizer_config
):
    param_groups = [
        {"params": tag_model.parameters(), "lr": tag_lr, "name": "tag_model"},
        {"params": bert_model.parameters(), "lr": bert_lr, "name": "bert_model"},
    ]

    if optimizer_name == "Adam":
        return optim.Adam(param_groups, **optimizer_config)
    elif optimizer_name == "AdamW":
        return optim.AdamW(param_groups, **optimizer_config)
    elif optimizer_name == "SGD":
        return optim.SGD(param_groups, **optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
