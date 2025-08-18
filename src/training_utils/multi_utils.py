import torch.nn as nn
import torch.optim as optim

from src.training_utils.datasets import JointCardDataset
from src.training_utils.generic_training_utils import (
    print_separator,
    print_models_param_summary,
    get_loss_tag_fn,
)


def build_training_components_multitask(
    config,
    bert_model,
    synergy_model,
    device,
    tag_model,
    tag_projector_model=None,
    tag_model_pos_weight=None,
    synergy_model_pos_weight=1.0,
    use_multitask_projector=False,
    multitask_projector_model=None,
    use_tag_projector=False,
):
    optimizer = build_multitask_optimizer(
        bert_model=bert_model,
        synergy_model=synergy_model,
        tag_projector_model=tag_projector_model,
        tag_model=tag_model,
        bert_lr=config["bert_learning_rate_multi"],
        synergy_lr=config["synergy_learning_rate"],
        tag_projector_lr=config["tag_projector_learning_rate"],
        tag_lr=config["tag_learning_rate_multi"],
        optimizer_config=config.get("optimizer_config", {}),
        optimizer_name=config.get("optimizer"),
        use_multitask_projector=use_multitask_projector,
        multitask_projector_model=multitask_projector_model,
        multitasak_proj_lr=config.get("multitask_projector_learning_rate_multi", None),
        use_tag_projector=use_tag_projector,
    )

    models_with_names = [
        ("bert_model", bert_model),
        ("synergy_model", synergy_model),
        ("tag_model", tag_model),
    ]

    if use_multitask_projector:
        models_with_names.append(
            ("multitask_projector_model", multitask_projector_model)
        )
    if use_tag_projector:
        models_with_names.append(("tag_projector_model", tag_projector_model))

    print_models_param_summary(models_with_names, optimizer)

    print_separator()

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=synergy_model_pos_weight).to(device)

    print(
        "Using BCEWithLogitsLoss for synergy model with pos_weight:",
        synergy_model_pos_weight,
    )

    print_separator()

    loss_tag_fn = get_loss_tag_fn(
        config=config, device=device, tag_model_pos_weight=tag_model_pos_weight
    )

    return optimizer, loss_fn, loss_tag_fn


def build_multitask_optimizer(
    optimizer_name,
    bert_model,
    synergy_model,
    tag_projector_model,
    bert_lr,
    tag_projector_lr,
    synergy_lr,
    optimizer_config,
    tag_model,
    tag_lr,
    use_multitask_projector=False,
    multitask_projector_model=None,
    multitasak_proj_lr=None,
    use_tag_projector=False,
):
    param_groups = [
        {"params": bert_model.parameters(), "lr": bert_lr, "name": "bert_model"},
        {
            "params": synergy_model.parameters(),
            "lr": synergy_lr,
            "name": "synergy_model",
        },
        {"params": tag_model.parameters(), "lr": tag_lr, "name": "tag_model"},
    ]

    if use_multitask_projector:
        param_groups.append(
            {
                "params": multitask_projector_model.parameters(),
                "lr": multitasak_proj_lr,
                "name": "multitask_projector_model",
            }
        )
    if use_tag_projector:
        param_groups.append(
            {
                "params": tag_projector_model.parameters(),
                "lr": tag_projector_lr,
                "name": "tag_projector_model",
            }
        )

    if optimizer_name == "Adam":
        return optim.Adam(param_groups, **optimizer_config)
    elif optimizer_name == "AdamW":
        return optim.AdamW(param_groups, **optimizer_config)
    elif optimizer_name == "SGD":
        return optim.SGD(param_groups, **optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_dataloaders_multi(config, tokenizer, index_splits):

    # Create DataLoaders
    dataloaders = {}
    for split_name, indices in index_splits.items():
        dataset = JointCardDataset(
            config["synergy_file"],
            config["bulk_file"],
            tokenizer,
            max_length=config["max_length_bert_tokenizer"],
            tags_len=config.get("tag_output_dim", None),
            subset_indices=indices,
            dataset_name=split_name,
            debug_dataset=True,
            tag_to_index_file=config.get("tag_to_index_file", None),
        )

        is_train = split_name == "train"
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=is_train,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2 if is_train else None,
        )

    return dataloaders
