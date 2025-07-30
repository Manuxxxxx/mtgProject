import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import sys
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from src.models.tag_model import build_tag_model
from src.models.bert_model import build_bert_model
from src.models.synergy_model import (
    build_synergy_model,
    calculate_synergy_weighted_FP_loss,
)
from src.models.tag_projector_model import build_tag_projector_model

from src.training_utils.generic_training_utils import (
    set_color,
    set_seed,
    print_separator,
    get_real_fake_indices,
    print_models_param_summary,
    split_indices,
)
from src.training_utils.loss_scaler import LossScaler
from src.training_utils.datasets import CardDataset
from src.training_utils.metrics import (
    log_metrics_multitask,
    log_metrics_tag,
    setup_dirs_writer,
    update_metrics_multi,
)
from src.training_utils.multi_utils import (
    build_training_components_multitask,
    create_dataloaders_multi,
)
from src.training_utils.tag_utils import build_training_components_tag


def get_embeddings_and_tag_preds(
    bert_model, tag_model, tag_projector_model, batch, device
):
    input_ids1 = batch["input_ids1"].to(device)
    attention_mask1 = batch["attention_mask1"].to(device)
    input_ids2 = batch["input_ids2"].to(device)
    attention_mask2 = batch["attention_mask2"].to(device)

    embed1 = bert_model(input_ids1, attention_mask1)
    embed2 = bert_model(input_ids2, attention_mask2)

    tags_pred1 = tag_model(embed1)
    tags_pred2 = tag_model(embed2)

    preds_tag1 = torch.sigmoid(tags_pred1)
    preds_tag2 = torch.sigmoid(tags_pred2)

    projected_tag_embed1 = tag_projector_model(preds_tag1.detach())
    projected_tag_embed2 = tag_projector_model(preds_tag2.detach())

    return (
        embed1,
        embed2,
        tags_pred1,
        tags_pred2,
        preds_tag1,
        preds_tag2,
        projected_tag_embed1,
        projected_tag_embed2,
    )


def compute_synergy_loss(
    synergy_model,
    embed1,
    embed2,
    projected_tag_embed1,
    projected_tag_embed2,
    labels_synergy,
    loss_synergy_model,
    false_positive_penalty,
):
    logits_synergy = synergy_model(
        embed1, embed2, projected_tag_embed1, projected_tag_embed2
    )
    weighted_loss_synergy, preds_synergy, _ = calculate_synergy_weighted_FP_loss(
        logits_synergy,
        labels_synergy,
        loss_synergy_model,
        false_positive_penalty=false_positive_penalty,
    )
    return weighted_loss_synergy, preds_synergy


def train_tag_loop(
    bert_model,
    tag_model,
    dataloader,
    optimizer,
    loss_tag_model,
    epoch,
    writer,
    device,
    accumulation_steps=1,
    use_empty_cache=False,
    calc_metrics=False,
):
    tag_model.train()
    bert_model.train()
    total_tag_loss = 0.0
    if calc_metrics:
        all_preds_tag, all_labels_tag = [], []

    scaler = GradScaler()
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Train Tag")):
        tag_hot = batch["tag_hot"].to(device)

        with autocast(device_type="cuda"):
            embed = bert_model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            preds_tag = tag_model(embed)

            tag_loss = loss_tag_model(preds_tag, tag_hot)
            full_loss = tag_loss
            loss_scaled = full_loss / accumulation_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if use_empty_cache:
                torch.cuda.empty_cache()

        total_tag_loss += tag_loss.item()

        if calc_metrics:
            all_preds_tag.extend(preds_tag.detach().cpu().numpy())
            all_labels_tag.extend(tag_hot.cpu().numpy())

    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_tag_loss / len(dataloader)

    log_metrics_tag(
        writer,
        epoch,
        avg_loss,
        all_preds_tag if calc_metrics else [],
        all_labels_tag if calc_metrics else [],
        "Train",
    )


def eval_tag_loop(
    bert_model, tag_model, dataloader, loss_tag_model, epoch, writer, device
):
    tag_model.eval()
    bert_model.eval()

    total_tag_loss = 0.0
    all_preds_tag, all_labels_tag = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval Tag"):
            tag_hot = batch["tag_hot"].to(device)

            embed = bert_model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            preds_tag = tag_model(embed)

            tag_loss = loss_tag_model(preds_tag, tag_hot)
            total_tag_loss += tag_loss.item()

            all_preds_tag.extend(preds_tag.cpu().numpy())
            all_labels_tag.extend(tag_hot.cpu().numpy())

    avg_loss = total_tag_loss / len(dataloader)

    log_metrics_tag(writer, epoch, avg_loss, all_preds_tag, all_labels_tag, "Val")


def train_multitask_loop(
    bert_model,
    synergy_model,
    tag_model,
    tag_projector_model,
    dataloader,
    optimizer,
    loss_synergy_model,
    loss_tag_model,
    epoch,
    writer,
    device,
    false_positive_penalty=1.0,
    tag_loss_weight=1,
    accumulation_steps=1,
    use_empty_cache=False,
    calc_metrics=False,
    scheduler=None,
):
    bert_model.train()
    synergy_model.train()
    tag_model.train()
    total_synergy_loss = 0.0
    total_tag_loss = 0.0
    if calc_metrics:
        all_preds_synergy, all_labels_synergy = [], []
        all_preds_tag, all_labels_tag = [], []

    scaler = GradScaler()
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Train")):
        labels_synergy = batch["label"].to(device)
        tag_hot1 = batch["tag_hot1"].to(device)
        tag_hot2 = batch["tag_hot2"].to(device)

        with autocast(device_type="cuda"):
            (
                embed1,
                embed2,
                tags_pred1,
                tags_pred2,
                preds_tag1,
                preds_tag2,
                projected_tag_embed1,
                projected_tag_embed2,
            ) = get_embeddings_and_tag_preds(
                bert_model, tag_model, tag_projector_model, batch, device
            )

            tag_loss1 = loss_tag_model(tags_pred1, tag_hot1)
            tag_loss2 = loss_tag_model(tags_pred2, tag_hot2)
            tag_loss = (tag_loss1 + tag_loss2) / 2.0
            weighted_loss_synergy, preds_synergy = compute_synergy_loss(
                synergy_model,
                embed1,
                embed2,
                projected_tag_embed1,
                projected_tag_embed2,
                labels_synergy,
                loss_synergy_model,
                false_positive_penalty,
            )

            # full_loss = weighted_loss_synergy + tag_loss_weight * tag_loss
            loss_scaler = LossScaler()
            full_loss, scale_factor = loss_scaler.get_scaled_total_loss(
                tag_loss, weighted_loss_synergy, alpha=tag_loss_weight
            )
            loss_scaled = full_loss / accumulation_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if use_empty_cache:
                torch.cuda.empty_cache()

        total_synergy_loss += weighted_loss_synergy.item()
        total_tag_loss += tag_loss.item() * tag_loss_weight

        if calc_metrics:
            update_metrics_multi(
                all_preds_synergy,
                all_labels_synergy,
                all_preds_tag,
                all_labels_tag,
                preds_synergy,
                labels_synergy,
                preds_tag1,
                preds_tag2,
                tag_hot1,
                tag_hot2,
            )

    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    avg_loss = (total_tag_loss + total_synergy_loss) / len(dataloader)
    avg_synergy_loss = total_synergy_loss / len(dataloader)
    avg_tag_loss = total_tag_loss / len(dataloader)

    log_metrics_multitask(
        writer,
        epoch,
        avg_loss,
        avg_synergy_loss,
        avg_tag_loss,
        all_preds_synergy if calc_metrics else [],
        all_labels_synergy if calc_metrics else [],
        all_preds_tag if calc_metrics else [],
        all_labels_tag if calc_metrics else [],
        "Train",
    )


def eval_multitask_loop(
    bert_model,
    synergy_model,
    tag_model,
    tag_projector_model,
    dataloader,
    loss_synergy_model,
    loss_tag_model,
    epoch,
    writer,
    device,
    label="Val",
    false_positive_penalty=1.0,
    tag_loss_weight=1.0,
):
    bert_model.eval()
    synergy_model.eval()
    tag_model.eval()

    total_synergy_loss = 0.0
    total_tag_loss = 0.0
    all_preds_synergy, all_labels_synergy = [], []
    all_preds_tag, all_labels_tag = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{label} Eval"):
            labels_synergy = batch["label"].to(device)
            tag_hot1 = batch["tag_hot1"].to(device)
            tag_hot2 = batch["tag_hot2"].to(device)

            (
                embed1,
                embed2,
                tags_pred1,
                tags_pred2,
                preds_tag1,
                preds_tag2,
                projected_tag_embed1,
                projected_tag_embed2,
            ) = get_embeddings_and_tag_preds(
                bert_model, tag_model, tag_projector_model, batch, device
            )

            tag_loss1 = loss_tag_model(tags_pred1, tag_hot1)
            tag_loss2 = loss_tag_model(tags_pred2, tag_hot2)
            tag_loss = (tag_loss1 + tag_loss2) / 2.0
            weighted_loss_synergy, preds_synergy = compute_synergy_loss(
                synergy_model,
                embed1,
                embed2,
                projected_tag_embed1,
                projected_tag_embed2,
                labels_synergy,
                loss_synergy_model,
                false_positive_penalty,
            )

            total_synergy_loss += weighted_loss_synergy.item()
            total_tag_loss += tag_loss.item() * tag_loss_weight

            update_metrics_multi(
                all_preds_synergy,
                all_labels_synergy,
                all_preds_tag,
                all_labels_tag,
                preds_synergy,
                labels_synergy,
                preds_tag1,
                preds_tag2,
                tag_hot1,
                tag_hot2,
            )

    avg_loss = (total_synergy_loss + total_tag_loss) / len(dataloader)
    avg_synergy_loss = total_synergy_loss / len(dataloader)
    avg_tag_loss = total_tag_loss / len(dataloader)

    log_metrics_multitask(
        writer,
        epoch,
        avg_loss,
        avg_synergy_loss,
        avg_tag_loss,
        all_preds_synergy,
        all_labels_synergy,
        all_preds_tag,
        all_labels_tag,
        label,
    )


def train_tag_model(
    config, writer, save_full_dir, bert_model, tag_model, tokenizer, device, start_epoch
):

    set_color("blue")

    bert_model.unfreeze_bert()

    # --- Freeze BERT based on config ---
    freeze_epochs = config.get("freeze_bert_epochs_tag", None)
    freeze_layers = config.get("freeze_bert_layers_tag", None)

    if freeze_layers == "all":
        print(
            f"Freezing all BERT layers for tag model training, for {freeze_epochs} epochs."
        )
        bert_model.freeze_bert()
    elif isinstance(freeze_layers, int):
        print(
            f"Freezing the first {freeze_layers} BERT layers for tag model training, for {freeze_epochs} epochs."
        )
        bert_model.freeze_bert_layers(freeze_layers)

    with open(config["bulk_file"], "r") as f:
        bulk_data = json.load(f)

    splits = config.get("splits_tag", {"train": 0.4, "val": 0.1})
    print("Using splits:", splits)

    random.shuffle(bulk_data)
    train_data = bulk_data[: int(len(bulk_data) * splits["train"])]
    val_data = bulk_data[
        int(len(bulk_data) * splits["train"]) : int(
            len(bulk_data) * (splits["train"] + splits["val"])
        )
    ]

    train_dataset = CardDataset(
        train_data,
        tokenizer,
        max_length=config["max_length_bert_tokenizer"],
        tags_len=config.get("tag_output_dim", None),
        dataset_name="train_tag",
        tag_to_index_file=config.get("tag_to_index_file", None),
    )
    val_dataset = CardDataset(
        val_data,
        tokenizer,
        max_length=config["max_length_bert_tokenizer"],
        tags_len=config.get("tag_output_dim", None),
        dataset_name="val_tag",
        tag_to_index_file=config.get("tag_to_index_file", None),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    print(f"Using tag model with output dimension: {config.get('tag_output_dim', 271)}")
    print_separator()

    if config.get("tag_model_pos_weight", None) is not None:
        tag_model_pos_weight = torch.tensor([config["tag_model_pos_weight"]]).to(device)
    else:
        tag_counts = train_dataset.tag_counts  # shape: (tags_len,)
        total = train_dataset.total_tag_samples  # scalar

        neg_counts = total - tag_counts  # how many times each tag is not present
        tag_model_pos_weight = (neg_counts / (tag_counts + 1e-6)).to(
            device
        )  # avoid div-by-zero

    optimizer, loss_tag_fn = build_training_components_tag(
        config, bert_model, tag_model, device, tag_model_pos_weight=tag_model_pos_weight
    )

    print_separator()
    end_epoch = config.get("epochs_tag", 0) + start_epoch
    print(f"Starting training for tag model for {config['epochs_tag']} epochs...\n")
    print(f"starting epoch: {start_epoch}, to epoch: {end_epoch}")
    if end_epoch <= start_epoch:
        print("No epochs to train for tag model, exiting.")
        return

    for epoch in tqdm(
        range(start_epoch, end_epoch),
        desc="Epochs Tag",
        initial=start_epoch,
        unit="epoch",
    ):
        # Unfreeze BERT when freeze period is over
        if freeze_epochs and epoch - start_epoch == freeze_epochs:
            print(f"Unfreezing BERT at epoch {epoch}")
            bert_model.unfreeze_bert()
            print_models_param_summary(
                [("bert_model", bert_model), ("tag_model", tag_model)], optimizer
            )
            print_separator()

        train_tag_loop(
            bert_model,
            tag_model,
            train_loader,
            optimizer,
            loss_tag_fn,
            epoch,
            writer,
            device,
            accumulation_steps=config["accumulation_steps"],
            use_empty_cache=config.get("use_empty_cache", False),
            calc_metrics=config.get("train_calc_metrics", False),
        )

        if epoch % config.get("save_every_n_epochs", 1) == 0:

            torch.save(
                bert_model.state_dict(),
                os.path.join(save_full_dir, f"bert_tag_only_epoch_{epoch + 1}.pt"),
            )
            torch.save(
                tag_model.state_dict(),
                os.path.join(save_full_dir, f"tag_tag_only_model_epoch_{epoch + 1}.pt"),
            )
            print(f"Saved Bert and Tag models at epoch {epoch + 1}.")

        print_separator()
        if epoch % config.get("eval_every_n_epochs", 1) == 0:
            eval_tag_loop(
                bert_model, tag_model, val_loader, loss_tag_fn, epoch, writer, device
            )

    print_separator()
    print("Training completed for tag model.")
    print("Saving final models...")
    torch.save(
        bert_model.state_dict(), os.path.join(save_full_dir, "bert_tag_only_final.pt")
    )
    torch.save(
        tag_model.state_dict(),
        os.path.join(save_full_dir, "tag_tag_only_final_model.pt"),
    )
    print("Final models saved.")
    print_separator()


def train_multitask_model(
    config, writer, save_full_dir, start_epoch, bert_model, tokenizer, device, tag_model
):
    set_color("green")

    bert_model.unfreeze_bert()  # Ensure BERT is unfrozen for multitask training

    print_separator()
    print("Starting Multitask Model Training")
    print_separator()

    # Load and split indices
    real_indices, fake_indices = get_real_fake_indices(config["synergy_file"])

    # Freeze BERT based on config
    freeze_epochs = config.get("freeze_bert_epochs_multi", None)
    freeze_layers = config.get("freeze_bert_layers_multi", None)

    if freeze_layers == "all":
        print(
            f"Freezing all BERT layers for tag model training, for {freeze_epochs} epochs."
        )
        bert_model.freeze_bert()
    elif isinstance(freeze_layers, int):
        print(
            f"Freezing the first {freeze_layers} BERT layers for tag model training, for {freeze_epochs} epochs."
        )
        bert_model.freeze_bert_layers(freeze_layers)

    # Define split proportions
    if config.get("splits", None) is not None:
        splits = config["splits"]
    else:
        print("Using default splits")
        splits = {
            "train": {"real": 0.8, "fake": 0.1},
            "val_real": {"real": 0.1, "fake": 0.0},
            "val_real_fake": {"real": 0.1, "fake": 0.03},
        }

    split_indices_result = split_indices(
        real_indices, fake_indices, splits, log_splits=True
    )

    print_separator()

    data_loaders = create_dataloaders_multi(config, tokenizer, split_indices_result)

    print_separator()

    train_loader = data_loaders["train"]

    synergy_model = build_synergy_model(
        config["synergy_arch"],
        config["bert_embedding_dim"],
        config["tag_projector_output_dim"],
    ).to(device)

    print(f"Using synergy model architecture: {config['synergy_arch']}")

    tag_projector_model = build_tag_projector_model(
        num_tags=config.get("tag_output_dim"),
        output_dim=config.get("tag_projector_output_dim"),
        hidden_dim=config.get("tag_projector_hidden_dim"),
        dropout=config.get("tag_projector_dropout"),
    ).to(device)

    print(
        f"Using tag projector model with output dimension: {config.get('tag_projector_output_dim')}"
    )

    if config.get("synergy_checkpoint", None) and config["synergy_checkpoint"] != "":
        synergy_model.load_state_dict(torch.load(config["synergy_checkpoint"]))
        print(f"Loaded synergy model checkpoint: {config['synergy_checkpoint']}")

    if (
        config.get("tag_projector_checkpoint", None)
        and config["tag_projector_checkpoint"] != ""
    ):
        tag_projector_model.load_state_dict(
            torch.load(config["tag_projector_checkpoint"])
        )
        print(
            f"Loaded tag projector model checkpoint: {config['tag_projector_checkpoint']}"
        )

    train_dataset = train_loader.dataset

    if config.get("synergy_pos_weight", None) is not None:
        synergy_model_pos_weight = torch.tensor([config["synergy_pos_weight"]])
        print(
            f"Using provided synergy model pos weight: {synergy_model_pos_weight.item()}"
        )
    else:
        synergy_1_counts = train_dataset.counts[0] + train_dataset.counts[2]
        synergy_0_counts = train_dataset.counts[1] + train_dataset.counts[3]

        synergy_model_pos_weight = torch.tensor(
            [synergy_0_counts / (synergy_1_counts + 1e-6)]
        )
        print(f"Calculated synergy model pos weight: {synergy_model_pos_weight.item()}")

    if config.get("tag_model_pos_weight", None) is not None:
        print(f"Using provided tag model pos weight: {config['tag_model_pos_weight']}")
        tag_model_pos_weight = torch.tensor([config["tag_model_pos_weight"]]).to(device)
    else:
        tag_counts = train_dataset.tag_counts  # shape: (tags_len,)
        total = train_dataset.total_tag_samples  # scalar

        neg_counts = total - tag_counts  # how many times each tag is not present
        tag_model_pos_weight = (neg_counts / (tag_counts + 1e-6)).to(
            device
        )  # avoid div-by-zero
        print(f"Calculated tag model pos weight: {tag_model_pos_weight}")

    optimizer, loss_sin_fn, loss_tag_fn = build_training_components_multitask(
        config=config,
        bert_model=bert_model,
        synergy_model=synergy_model,
        tag_model=tag_model,
        tag_projector_model=tag_projector_model,
        device=device,
        tag_model_pos_weight=tag_model_pos_weight,
        synergy_model_pos_weight=synergy_model_pos_weight,
    )

    if config.get("use_scheduler", False):
        num_training_steps = len(train_loader) * (config["epochs_multi"] - start_epoch)
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(
            f"Using scheduler with {num_warmup_steps} warmup steps and {num_training_steps} total steps."
        )
    else:
        scheduler = None
        print("Not using scheduler.")

    print_separator()

    test_at_start_sets = config.get("test_at_start_sets", None)
    if (
        test_at_start_sets is not None
        and isinstance(test_at_start_sets, list)
        and len(test_at_start_sets) > 0
    ):
        print_separator()
        print("Running initial evaluation on test sets...")
        for split_name, loader in data_loaders.items():
            if split_name in test_at_start_sets:
                eval_multitask_loop(
                    bert_model=bert_model,
                    synergy_model=synergy_model,
                    tag_model=tag_model,
                    tag_projector_model=tag_projector_model,
                    dataloader=loader,
                    loss_synergy_model=loss_sin_fn,
                    loss_tag_model=loss_tag_fn,
                    epoch=-1,  # No epoch for initial eval
                    writer=writer,
                    device=device,
                    label=split_name,
                    false_positive_penalty=config.get("synergy_false_positive_penalty"),
                    tag_loss_weight=config.get("tag_loss_weight"),
                )

    end_epoch = start_epoch + config.get("epochs_multi", 0)
    print_separator()
    print(f"Starting training for multitask model {config['epochs_multi']} epochs...\n")
    print(f"starting epoch: {start_epoch}, to epoch: {end_epoch}")
    for epoch in tqdm(
        range(start_epoch, end_epoch), desc="Epochs", initial=start_epoch
    ):
        print_separator()

        if freeze_epochs and epoch - start_epoch == freeze_epochs:
            print(f"Unfreezing BERT at epoch {epoch}")
            bert_model.unfreeze_bert()
            print_models_param_summary(
                [
                    ("bert_model", bert_model),
                    ("synergy_model", synergy_model),
                    ("tag_model", tag_model),
                    ("tag_projector_model", tag_projector_model),
                ],
                optimizer,
            )

        train_multitask_loop(
            bert_model=bert_model,
            synergy_model=synergy_model,
            tag_model=tag_model,
            tag_projector_model=tag_projector_model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_synergy_model=loss_sin_fn,
            loss_tag_model=loss_tag_fn,
            epoch=epoch,
            writer=writer,
            device=device,
            false_positive_penalty=config.get("synergy_false_positive_penalty"),
            tag_loss_weight=config.get("tag_loss_weight"),
            accumulation_steps=config["accumulation_steps"],
            use_empty_cache=config.get("use_empty_cache", False),
            calc_metrics=config.get("train_calc_metrics", False),
            scheduler=scheduler,
        )

        if (epoch + 1) % config["save_every"] == 0:
            print_separator()
            bert_model_path = os.path.join(
                save_full_dir, f"bert_multi_model_epoch_{epoch + 1}.pth"
            )
            synergy_model_path = os.path.join(
                save_full_dir, f"synergy_model_epoch_{epoch + 1}.pth"
            )
            tag_model_path = os.path.join(
                save_full_dir, f"tag_multi_model_epoch_{epoch + 1}.pth"
            )
            tag_projector_model_path = os.path.join(
                save_full_dir, f"tag_projector_model_epoch_{epoch + 1}.pth"
            )

            torch.save(tag_projector_model.state_dict(), tag_projector_model_path)
            torch.save(tag_model.state_dict(), tag_model_path)
            torch.save(bert_model.state_dict(), bert_model_path)
            torch.save(synergy_model.state_dict(), synergy_model_path)
            print(
                f"Saved Bert, Synergy, Tag, and Tag Projector models at epoch {epoch + 1}."
            )

        if (epoch + 1) % config["eval_every"] == 0:
            print_separator()
            for split_name, loader in data_loaders.items():
                if split_name.startswith("val"):
                    eval_multitask_loop(
                        bert_model=bert_model,
                        synergy_model=synergy_model,
                        tag_model=tag_model,
                        tag_projector_model=tag_projector_model,
                        dataloader=loader,
                        loss_synergy_model=loss_sin_fn,
                        loss_tag_model=loss_tag_fn,
                        epoch=epoch,
                        writer=writer,
                        device=device,
                        label=split_name,
                        false_positive_penalty=config.get(
                            "synergy_false_positive_penalty"
                        ),
                        tag_loss_weight=config.get("tag_loss_weight"),
                    )

    # Final save
    bert_model_path = os.path.join(save_full_dir, "bert_model_final.pth")
    synergy_model_path = os.path.join(save_full_dir, "synergy_model_final.pth")
    tag_model_path = os.path.join(save_full_dir, "tag_model_final.pth")
    tag_projector_model_path = os.path.join(
        save_full_dir, "tag_projector_model_final.pth"
    )
    torch.save(bert_model.state_dict(), bert_model_path)
    torch.save(synergy_model.state_dict(), synergy_model_path)
    torch.save(tag_model.state_dict(), tag_model_path)
    torch.save(tag_projector_model.state_dict(), tag_projector_model_path)
    print(f"Final models saved at {save_full_dir}")
    writer.close()


def run_training_multitask(config):
    """
    Main function to run the training of the multitask model.
    It sets up directories, loads data, initializes models, and starts training.
    """
    print_separator()
    print("Starting Multitask Training")
    print_separator()

    # Set up directories and writer
    writer, save_full_dir, start_epoch = setup_dirs_writer(config)

    # Initialize BERT model and tokenizer
    print("Loading BERT model and tokenizer...")
    # Step 2: Initialize BERT model and tokenizer
    model_name = config["bert_model_name"]
    embedding_dim = config.get("bert_embedding_dim")
    bert_model, tokenizer, device = build_bert_model(model_name, embedding_dim)
    print(f"Using BERT model: {model_name} with embedding dimension: {embedding_dim}")

    tag_model = build_tag_model(
        "tagModel",
        input_dim=config.get("bert_embedding_dim"),
        output_dim=config.get("tag_output_dim"),
        hidden_dims=config.get("tag_hidden_dims"),
        dropout=config.get("tag_dropout"),
        use_batchnorm=config.get("tag_use_batchnorm"),
        use_sigmoid_output=config.get("tag_use_sigmoid_output"),
    ).to(device)

    # Train the tag model if specified
    if config.get("train_tag_model", False):

        print("Training tag model...")
        print_separator()
        if (
            config.get("bert_checkpoint_tag", None)
            and config["bert_checkpoint_tag"] != ""
        ):
            bert_model.load_state_dict(torch.load(config["bert_checkpoint_tag"]))
            print(
                f"Loaded BERT checkpoint for tag model: {config['bert_checkpoint_tag']}"
            )

        if (
            config.get("tag_checkpoint_tag", None)
            and config["tag_checkpoint_tag"] != ""
        ):
            tag_model.load_state_dict(torch.load(config["tag_checkpoint_tag"]))
            print(f"Loaded tag model checkpoint: {config['tag_checkpoint_tag']}")

        train_tag_model(
            config,
            writer,
            save_full_dir,
            bert_model,
            tag_model,
            tokenizer,
            device,
            start_epoch,
        )
    else:

        print("Skipping tag model training as per configuration.")

    if config["bert_checkpoint_multi"] and config["bert_checkpoint_multi"] != "":
        bert_model.load_state_dict(torch.load(config["bert_checkpoint_multi"]))
        print(f"Loaded BERT checkpoint: {config['bert_checkpoint_multi']}")

    if (
        config.get("tag_checkpoint_multi", None)
        and config["tag_checkpoint_multi"] != ""
    ):
        tag_model.load_state_dict(torch.load(config["tag_checkpoint_multi"]))
        print(f"Loaded tag model checkpoint: {config['tag_checkpoint_multi']}")

    train_multitask_model(
        config,
        writer,
        save_full_dir,
        start_epoch + config.get("epochs_tag", 0),
        bert_model,
        tokenizer,
        device,
        tag_model,
    )


def run_all_configs(config_path):
    with open(config_path) as f:
        config_list = json.load(f)

    for config in config_list:
        config["run_name"] += time.strftime("_%Y%m%d_%H%M%S")  # Append timestamp here

        run_training_multitask(config)


if __name__ == "__main__":
    set_seed(1006)
    if len(sys.argv) != 2:
        print("Usage: python joint_train.py <config_file.json>")
        exit(1)

    config_file = sys.argv[1]
    run_all_configs(config_file)
