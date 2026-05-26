import argparse
import time
from pathlib import Path

import torch
from torch import nn

from models import build_model, count_trainable_parameters, model_size_megabytes
from utils.config import ensure_output_directories, experiment_paths, load_config
from utils.data import build_dataloader, build_dataset
from utils.engine import evaluate, train_one_epoch
from utils.io import save_history, save_json
from utils.runtime import resolve_device
from utils.seed import set_seed
from utils.visualization import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FCN on the five-class PASCAL VOC subset.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration.")
    parser.add_argument("--download", action="store_true", help="Download VOC2012 if it is not present.")
    parser.add_argument("--resume", help="Checkpoint path from which training should resume.")
    parser.add_argument("--device", help="Override the configured device, for example cuda:0 or cpu.")
    return parser.parse_args()


def build_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    train_config = config["train"]
    name = train_config["optimizer"].lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=train_config["learning_rate"],
            momentum=train_config["momentum"],
            weight_decay=train_config["weight_decay"],
        )
    raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = experiment_paths(config)
    ensure_output_directories(paths)
    set_seed(config["runtime"]["seed"], config["runtime"]["deterministic"])
    device = resolve_device(args.device or config["runtime"]["device"])

    train_dataset = build_dataset(config, config["data"]["train_split"], is_train=True, download=args.download)
    val_dataset = build_dataset(config, config["data"]["val_split"], is_train=False, download=args.download)
    train_loader = build_dataloader(config, train_dataset, is_train=True)
    val_loader = build_dataloader(config, val_dataset, is_train=False)

    model = build_model(config["model"], pretrained=not bool(args.resume) and config["model"]["pretrained"])
    model.to(device)
    # Isolate data augmentation randomness from backbone-specific model construction.
    set_seed(config["runtime"]["seed"], config["runtime"]["deterministic"])
    optimizer = build_optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])
    criterion = nn.CrossEntropyLoss(ignore_index=config["data"]["ignore_index"])
    use_amp = config["train"]["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_miou = -1.0
    history: list[dict] = []
    elapsed_before_resume = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", best_miou)
        history = checkpoint.get("history", history)
        elapsed_before_resume = checkpoint.get("total_training_seconds", elapsed_before_resume)

    training_start = time.perf_counter()
    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch)
        val_result = evaluate(
            model,
            val_loader,
            criterion,
            device,
            config["model"]["num_classes"],
            config["data"]["ignore_index"],
        )
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "val_pixel_accuracy": val_result["pixel_accuracy"],
            "val_miou": val_result["miou"],
            "epoch_seconds": train_result["seconds"] + val_result["seconds"],
        }
        history.append(row)
        save_history(history, paths["history"])
        plot_training_history(history, paths["curve"])
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "best_miou": max(best_miou, val_result["miou"]),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "history": history,
            "total_training_seconds": elapsed_before_resume + time.perf_counter() - training_start,
        }
        torch.save(checkpoint, paths["checkpoint_last"])
        if val_result["miou"] > best_miou:
            best_miou = val_result["miou"]
            torch.save(checkpoint, paths["checkpoint_best"])
        print(
            f"Epoch {epoch:03d}: train_loss={train_result['loss']:.4f}, "
            f"val_loss={val_result['loss']:.4f}, "
            f"PA={val_result['pixel_accuracy'] * 100:.2f}%, "
            f"mIoU={val_result['miou'] * 100:.2f}%"
        )

    summary = {
        "experiment": config["experiment"]["name"],
        "backbone": config["model"]["backbone"],
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "parameters": count_trainable_parameters(model),
        "model_size_megabytes": model_size_megabytes(model),
        "best_miou": best_miou,
        "total_training_seconds": elapsed_before_resume + time.perf_counter() - training_start,
        "best_checkpoint": str(paths["checkpoint_best"]),
    }
    save_json(summary, paths["train_summary"])
    print(f"Best checkpoint: {Path(paths['checkpoint_best'])}")


if __name__ == "__main__":
    main()
