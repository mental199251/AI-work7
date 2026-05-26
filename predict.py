import argparse
from pathlib import Path

import torch
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
    parser.add_argument("--compare", action="store_true", help="Render the default ResNet18 and ResNet34 comparison.")
    parser.add_argument("--configs", nargs="+", help="One or more configurations to render together.")
    parser.add_argument("--checkpoints", nargs="+", help="Optional checkpoint paths matching --configs.")
    parser.add_argument("--indices", nargs="+", type=int, help="Validation subset indices to visualize.")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", help="Directory for generated panels.")
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
        foreground = {int(value) for value in torch.unique(target) if 0 < int(value) < dataset.num_classes}
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
    if args.configs:
        config_files = args.configs
    elif args.compare:
        config_files = DEFAULT_COMPARE_CONFIGS
    else:
        config_files = [args.config]
    configs = [load_config(path) for path in config_files]
    first_config = configs[0]
    device = resolve_device(args.device or first_config["runtime"]["device"])
    if any(config["data"]["input_size"] != first_config["data"]["input_size"] for config in configs):
        raise ValueError("Comparison visualization requires identical input sizes for both experiments.")
    if any(config["data"].get("task", "five_class") != first_config["data"].get("task", "five_class") for config in configs):
        raise ValueError("Comparison visualization requires configurations using the same label task.")

    dataset = build_dataset(first_config, first_config["data"]["val_split"], is_train=False)
    if args.indices:
        indices = args.indices
    else:
        indices = choose_representative_indices(dataset, args.num_samples)
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError(f"Requested sample index is outside the validation subset of length {len(dataset)}.")

    comparison = len(configs) > 1
    if args.checkpoints and len(args.checkpoints) != len(configs):
        raise ValueError("The number of checkpoints must match the number of configurations.")
    if args.checkpoints:
        checkpoint_paths = [Path(path) for path in args.checkpoints]
    elif comparison:
        checkpoint_paths = (
            [experiment_paths(config)["checkpoint_best"] for config in configs]
        )
    else:
        checkpoint_paths = [
            Path(args.checkpoint) if args.checkpoint else experiment_paths(first_config)["checkpoint_best"]
        ]
    if comparison:
        output_directory = Path(args.output_dir) if args.output_dir else Path(first_config["output"]["root"]) / "visualizations" / "comparison"
    else:
        output_directory = Path(args.output_dir) if args.output_dir else experiment_paths(first_config)["visualizations"]
    models = [
        load_trained_model(config, checkpoint_path, device)
        for config, checkpoint_path in zip(configs, checkpoint_paths)
    ]

    for index in indices:
        image, target = dataset[index]
        batch = image.unsqueeze(0).to(device)
        predictions = {
            config["experiment"]["name"]: model(batch).argmax(dim=1)[0].cpu()
            for config, model in zip(configs, models)
        }
        output_path = output_directory / f"sample_{index:04d}.png"
        save_prediction_panel(
            image,
            target,
            predictions,
            output_path,
            class_colors=dataset.class_colors,
            class_names=dataset.class_names,
        )
        print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
