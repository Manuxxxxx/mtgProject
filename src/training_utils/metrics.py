from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import shutil
import json

def log_metrics_multitask(writer, epoch, avg_loss, avg_synergy_loss, avg_tag_loss, all_preds_synergy, all_labels_synergy, all_preds_tag, all_labels_tag, avg_penalty_ancestors_loss, label_prefix="Train"):
    writer.add_scalar(f"{label_prefix}/Loss", avg_loss, epoch)
    writer.add_scalar(f"{label_prefix}/Synergy Loss", avg_synergy_loss, epoch)

    precision_synergy = precision_score(all_labels_synergy, all_preds_synergy, zero_division=0)
    recall_synergy = recall_score(all_labels_synergy, all_preds_synergy, zero_division=0)
    f1_synergy = f1_score(all_labels_synergy, all_preds_synergy, zero_division=0)
    cm_synergy = confusion_matrix(all_labels_synergy, all_preds_synergy)

    print(
        f"{label_prefix} [{epoch+1}]| Loss: {avg_loss:.4f}  "
        f"| Synergy Loss: {avg_synergy_loss:.4f} | Precision Synergy: {precision_synergy:.4f} | Recall Synergy: {recall_synergy:.4f} | F1 Synergy: {f1_synergy:.4f} |"
    )

    writer.add_scalar(f"{label_prefix}/Precision", precision_synergy, epoch)
    writer.add_scalar(f"{label_prefix}/Recall", recall_synergy, epoch)
    writer.add_scalar(f"{label_prefix}/F1", f1_synergy, epoch)
    writer.add_scalar(f"{label_prefix}_cmSin/TP", cm_synergy[1, 1], epoch)
    writer.add_scalar(f"{label_prefix}_cmSin/TN", cm_synergy[0, 0], epoch)
    writer.add_scalar(f"{label_prefix}_cmSin/FP", cm_synergy[0, 1], epoch)
    writer.add_scalar(f"{label_prefix}_cmSin/FN", cm_synergy[1, 0], epoch)

    log_metrics_tag(writer, epoch, avg_tag_loss, avg_penalty_ancestors_loss, all_preds_tag, all_labels_tag, label_prefix)

def log_metrics_tag(writer, epoch, avg_tag_loss, avg_penalty_ancestors_loss, all_preds_tag, all_labels_tag, label_prefix="Train"):
    writer.add_scalar(f"{label_prefix}/Tag Loss", avg_tag_loss, epoch)
    writer.add_scalar(f"{label_prefix}/Penalty Ancestors", avg_penalty_ancestors_loss, epoch)
    binary_preds_tag = np.array(all_preds_tag) > 0.5

    # print(f"shape of all_preds_tag: {np.array(all_preds_tag).shape}")
    # print(f"shape of all_labels_tag: {np.array(all_labels_tag).shape}")
    # # Convert to integers (True → 1, False → 0)
    # binary_preds_tag_int = binary_preds_tag.astype(int)
    # # Total number of 1s
    # total_ones = np.sum(binary_preds_tag_int)
    # total_ones_labels = np.sum(np.array(all_labels_tag))
    # # Total number of predictions (575 * 103)
    # total_preds = binary_preds_tag_int.size
    # # Total number of 0s
    # total_zeros = total_preds - total_ones
    # total_zeros_labels = len(all_labels_tag) * binary_preds_tag_int.shape[1] - total_ones_labels
    # # Average number of 1s per prediction (per row)
    # avg_ones_per_prediction = np.mean(np.sum(binary_preds_tag_int, axis=1))
    # avg_ones_per_labels = np.mean(np.sum(np.array(all_labels_tag), axis=1))
    # # Output results
    # print(f"Total 1s: {total_ones}, Total 0s: {total_zeros}")
    # print(f"Total 1s in labels: {total_ones_labels}, Total 0s in labels: {total_zeros_labels}")
    # print(f"Average 1s per prediction: {avg_ones_per_prediction:.2f}")
    # print(f"Average 1s per labels: {avg_ones_per_labels:.2f}")
    
    precision_tag = precision_score(all_labels_tag, binary_preds_tag, average='macro', zero_division=0)
    recall_tag = recall_score(all_labels_tag, binary_preds_tag, average='macro', zero_division=0)
    f1_tag = f1_score(all_labels_tag, binary_preds_tag, average='macro', zero_division=0)
    cm_tag = confusion_matrix(np.array(all_labels_tag).flatten(), binary_preds_tag.flatten())

    print(f"{label_prefix} [{epoch+1}]| Tag Loss: {avg_tag_loss:.4f} | Penalty Anc Loss: {avg_penalty_ancestors_loss:.4f} | Precision Tag: {precision_tag:.4f} | Recall Tag: {recall_tag:.4f} | F1 Tag: {f1_tag:.4f} |")

    writer.add_scalar(f"{label_prefix}_tag/Precision", precision_tag, epoch)
    writer.add_scalar(f"{label_prefix}_tag/Recall", recall_tag, epoch)
    writer.add_scalar(f"{label_prefix}_tag/F1", f1_tag, epoch)
    writer.add_scalar(f"{label_prefix}_tag/cmTag/TP", cm_tag[1, 1], epoch)
    writer.add_scalar(f"{label_prefix}_tag/cmTag/TN", cm_tag[0, 0], epoch)
    writer.add_scalar(f"{label_prefix}_tag/cmTag/FP", cm_tag[0, 1], epoch)
    writer.add_scalar(f"{label_prefix}_tag/cmTag/FN", cm_tag[1, 0], epoch)

def clone_content_of_dir(src_dir, dest_dir):
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def setup_dirs_writer(config):
    start_epoch = config.get("start_epoch", 0)
    if start_epoch is None:
        start_epoch = 0
    
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")
        # Load the last save model writer and logs
        previous_run_name = config["run_name_previous"]
        # clone the log directory
        print(f"Cloning previous run directories from {previous_run_name} to {config['run_name']}")
        prev_log_full_dir = os.path.join(config["log_dir"], previous_run_name)
        prev_save_full_dir = os.path.join(config["save_dir"], previous_run_name)
        if not os.path.exists(prev_save_full_dir) or not os.path.exists(prev_log_full_dir):
            raise FileNotFoundError(
                f"Previous run directories do not exist: {prev_save_full_dir} or {prev_log_full_dir}"
            )
        new_run_name = config["run_name"]
        log_full_dir = os.path.join(config["log_dir"], new_run_name)
        save_full_dir = os.path.join(config["save_dir"], new_run_name)
        os.makedirs(log_full_dir, exist_ok=True)
        os.makedirs(save_full_dir, exist_ok=True)
        clone_content_of_dir(prev_log_full_dir, log_full_dir)
        clone_content_of_dir(prev_save_full_dir, save_full_dir)
        writer = SummaryWriter(log_dir=log_full_dir)
        
        print(f"Loaded previous run logs from {prev_log_full_dir}")
    else:
        print("Starting from scratch, no previous epoch to resume.")

        log_full_dir = os.path.join(config["log_dir"], config["run_name"])
        save_full_dir = os.path.join(config["save_dir"], config["run_name"])

        os.makedirs(log_full_dir, exist_ok=True)
        os.makedirs(save_full_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=log_full_dir)

        # write var config with writer
        writer.add_text("Config", json.dumps(config, indent=4))
    
    return writer, save_full_dir, start_epoch

def update_metrics_multi(all_preds_synergy, all_labels_synergy, all_preds_tag, all_labels_tag, preds_synergy, labels_synergy, preds_tag1, preds_tag2, tag_hot1, tag_hot2):
    all_preds_synergy.extend(preds_synergy.cpu().numpy())
    all_labels_synergy.extend(labels_synergy.cpu().numpy().astype(int))
    all_preds_tag.extend(preds_tag1.detach().cpu().numpy())
    all_preds_tag.extend(preds_tag2.detach().cpu().numpy())
    all_labels_tag.extend(tag_hot1.cpu().numpy())
    all_labels_tag.extend(tag_hot2.cpu().numpy())
