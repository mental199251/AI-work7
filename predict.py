import argparse
from pathlib import Path

import torch

from datasets import CLASS_NAMES
from models import build_model
from utils.config import experiment_paths, load_config
from utils.data import build_dataset
from utils.runtime import resolve_device
from utils.visualization import save_prediction_panel


DEFAULT_COMPARE_CONFIGS = ["configs/fcn_resnet18.yaml", "configs/fcn_resnet34.yaml"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FCN segmentation visualizations.")
    parser.add_argument("--config", default=DEFAULT_COMPARE_CONFIGS[0], help="Configuration for single-model output.")
    parser.add_argument("--checkpoint", help="Checkpoint for single-model output; defaults to best checkpoint.")
    parser.add_argument("--compare", action="store_true", help="Render ResNet18 and ResNet34 predictions together.")
    parser.add_argument("--configs", nargs=2, default=DEFAULT_COMPARE_CONFIGS, help="Two configs used with --compare.")
    parser.add_argument("--checkpoints", nargs=2, help="Optional two checkpoint paths used with --compare.")
    parser.add_argument("--indices", nargs="+", type=int, help="Validation subset indices to visualize.")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", help="Override configured device.")
    return parser.parse_args()


def load_trained_model(config: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(config["model"], pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def choose_representative_indices(dataset, count: int) -> list[int]:
    selected: list[int] = []
    covered_classes: set[int] = set()
    for index in range(len(dataset)):
        _, target = dataset[index]
        foreground = {int(value) for value in torch.unique(target) if 0 < int(value) < len(CLASS_NAMES)}
        if foreground - covered_classes:
            selected.append(index)
            covered_classes.update(foreground)
        if len(selected) >= count:
            return selected
    for index in range(len(dataset)):
        if index not in selected:
            selected.append(index)
        if len(selected) >= count:
            break
    return selected


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config_files = args.configs if args.compare else [args.config]
    configs = [load_config(path) for path in config_files]
    first_config = configs[0]
    device = resolve_device(args.device or first_config["runtime"]["device"])
    if any(config["data"]["input_size"] != first_config["data"]["input_size"] for config in configs):
        raise ValueError("Comparison visualization requires identical input sizes for both experiments.")

    dataset = build_dataset(first_config, first_config["data"]["val_split"], is_train=False)
    if args.indices:
        indices = args.indices
    else:
        indices = choose_representative_indices(dataset, args.num_samples)
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError(f"Requested sample index is outside the validation subset of length {len(dataset)}.")

    if args.compare:
        checkpoint_paths = (
            [Path(path) for path in args.checkpoints]
            if args.checkpoints
            else [experiment_paths(config)["checkpoint_best"] for config in configs]
        )
        output_directory = Path(first_config["output"]["root"]) / "visualizations" / "comparison"
    else:
        checkpoint_paths = [
            Path(args.checkpoint) if args.checkpoint else experiment_paths(first_config)["checkpoint_best"]
        ]
        output_directory = experiment_paths(first_config)["visualizations"]
    models = [
        load_trained_model(config, checkpoint_path, device)
        for config, checkpoint_path in zip(configs, checkpoint_paths)
    ]

    for index in indices:
        image, target = dataset[index]
        batch = image.unsqueeze(0).to(device)
        predictions = {
            f"FCN-{config['model']['backbone'].replace('resnet', 'ResNet')}": model(batch).argmax(dim=1)[0].cpu()
            for config, model in zip(configs, models)
        }
        output_path = output_directory / f"sample_{index:04d}.png"
        save_prediction_panel(image, target, predictions, output_path)
        print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
