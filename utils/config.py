from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["_config_path"] = str(config_path)
    return config


def experiment_paths(config: dict) -> dict[str, Path]:
    output_root = Path(config["output"]["root"])
    name = config["experiment"]["name"]
    return {
        "root": output_root,
        "checkpoint_best": output_root / "checkpoints" / f"{name}_best.pth",
        "checkpoint_last": output_root / "checkpoints" / f"{name}_last.pth",
        "history": output_root / "logs" / f"{name}_history.csv",
        "train_summary": output_root / "metrics" / f"{name}_train_summary.json",
        "evaluation": output_root / "metrics" / f"{name}_evaluation.json",
        "curve": output_root / "visualizations" / f"{name}_curves.png",
        "visualizations": output_root / "visualizations" / name,
    }


def ensure_output_directories(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        directory = path if key in {"root", "visualizations"} else path.parent
        directory.mkdir(parents=True, exist_ok=True)
