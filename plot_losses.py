#!/usr/bin/env python3
"""
Plot training/validation loss curves for NHITS and TimesNet from Lightning event logs.
Outputs: results/loss_curves.png
"""
import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

plt.style.use("seaborn-v0_8-whitegrid")

LOGS = {
    "NHITS": "lightning_logs",
    "TIMESNET": "lightning_logs_timesnet",
}

PREFERRED_TAGS = ["val_loss", "validation_loss", "valid_loss", "loss/val", "val/loss"]
TRAIN_TAGS = ["train_loss", "training_loss", "loss/train", "train/loss", "loss"]


def latest_event_file(log_dir: str):
    # Pick the newest events file across versions
    pattern = os.path.join(log_dir, "**", "events.*")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_scalars(event_file: str):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    def first_available(candidates):
        for t in candidates:
            if t in tags:
                return t
        return None

    val_tag = first_available(PREFERRED_TAGS)
    train_tag = first_available(TRAIN_TAGS)

    data = {}
    if train_tag:
        train_events = ea.Scalars(train_tag)
        data["train"] = ([e.step for e in train_events], [e.value for e in train_events])
    if val_tag:
        val_events = ea.Scalars(val_tag)
        data["val"] = ([e.step for e in val_events], [e.value for e in val_events])
    return data, val_tag, train_tag


def plot_model(ax, name, data, val_tag, train_tag):
    ax.set_title(f"{name} loss")
    if "train" in data:
        ax.plot(data["train"][0], data["train"][1], label=f"train ({train_tag})")
    if "val" in data:
        ax.plot(data["val"][0], data["val"][1], label=f"val ({val_tag})")
    if not data:
        ax.text(0.5, 0.5, "No loss data found", ha="center", va="center")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()


def main():
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (model, log_dir) in zip(axes, LOGS.items()):
        event_file = latest_event_file(log_dir)
        if not event_file:
            ax.text(0.5, 0.5, f"No events in {log_dir}", ha="center", va="center")
            ax.set_title(f"{model} loss")
            ax.axis("off")
            continue
        data, val_tag, train_tag = load_scalars(event_file)
        plot_model(ax, model, data, val_tag, train_tag)

    plt.tight_layout()
    out = "results/loss_curves.png"
    plt.savefig(out, dpi=150)
    print(f"Saved loss curves to {out}")


if __name__ == "__main__":
    main()
