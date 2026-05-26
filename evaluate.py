import argparse
from pathlib import Path

import torch
from torch import nn

from models import build_model, count_trainable_parameters, model_size_megabytes
from utils.benchmark import benchmark_single_image_inference
from utils.config import ensure_output_directories, experiment_paths, load_config
from utils.data import build_dataloader, build_dataset
from utils.engine import evaluate
from utils.io import save_json
from utils.runtime import resolve_device
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FCN checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Defaults to the best checkpoint defined by the config.")
    parser.add_argument("--device", help="Override configured device.")
    parser.add_argument("--benchmark-images", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = experiment_paths(config)
    ensure_output_directories(paths)
    device = resolve_device(args.device or config["runtime"]["device"])
    set_seed(config["runtime"]["seed"], config["runtime"]["deterministic"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else paths["checkpoint_best"]

    dataset = build_dataset(config, config["data"]["val_split"], is_train=False)
    loader = build_dataloader(config, dataset, is_train=False)
    model = build_model(config["model"], pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss(ignore_index=config["data"]["ignore_index"])

    result = evaluate(
        model,
        loader,
        criterion,
        device,
        config["model"]["num_classes"],
        config["data"]["ignore_index"],
    )
    result.update(
        benchmark_single_image_inference(
            model,
            dataset,
            device,
            warmup_iterations=config["evaluation"]["warmup_iterations"],
            measured_images=args.benchmark_images,
        )
    )
    result.update(
        {
            "experiment": config["experiment"]["name"],
            "backbone": config["model"]["backbone"],
            "checkpoint": str(checkpoint_path),
            "parameters": count_trainable_parameters(model),
            "model_size_megabytes": model_size_megabytes(model),
            "device": str(device),
        }
    )
    save_json(result, paths["evaluation"])
    print(f"Pixel Accuracy: {result['pixel_accuracy'] * 100:.2f}%")
    print(f"mIoU: {result['miou'] * 100:.2f}%")
    for name, iou in result["class_iou"].items():
        value = "N/A" if iou is None else f"{iou * 100:.2f}%"
        print(f"{name:>10}: {value}")
    print(f"Single-image inference: {result['single_image_milliseconds']:.2f} ms")
    print(f"Metrics saved to: {paths['evaluation']}")


if __name__ == "__main__":
    main()
