import argparse
import csv
from pathlib import Path

from datasets import get_task_spec
from utils.config import experiment_paths, load_config
from utils.io import load_json


DEFAULT_CONFIGS = ["configs/fcn_resnet18.yaml", "configs/fcn_resnet34.yaml"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation results from both FCN experiments.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--summary-stem", default="model_comparison")
    parser.add_argument("--classes-stem", default="class_iou_comparison")
    parser.add_argument("--title", default="Model Comparison")
    return parser.parse_args()


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}"


def main() -> None:
    args = parse_args()
    rows = []
    class_rows = []
    tasks = set()
    for config_file in args.configs:
        config = load_config(config_file)
        task = config["data"].get("task", "five_class")
        tasks.add(task)
        class_names = get_task_spec(task).class_names
        paths = experiment_paths(config)
        if not paths["evaluation"].exists():
            raise FileNotFoundError(
                f"Evaluation metrics missing: {paths['evaluation']}. Run evaluate.py for each model first."
            )
        evaluation = load_json(paths["evaluation"])
        training = load_json(paths["train_summary"]) if paths["train_summary"].exists() else {}
        rows.append(
            {
                "Model": config["experiment"]["name"],
                "Backbone": config["model"]["backbone"],
                "Upsampling": config["model"].get("upsampling", "bilinear"),
                "Task": task,
                "Parameters (M)": f"{evaluation['parameters'] / 1e6:.3f}",
                "Model Size (MB)": f"{evaluation['model_size_megabytes']:.2f}",
                "Pixel Accuracy (%)": percent(evaluation["pixel_accuracy"]),
                "mIoU (%)": percent(evaluation["miou"]),
                "Inference (ms/image)": f"{evaluation['single_image_milliseconds']:.2f}",
                "Training Time (s)": (
                    f"{training['total_training_seconds']:.2f}" if "total_training_seconds" in training else "N/A"
                ),
            }
        )
        class_rows.append(
            {
                "Model": config["experiment"]["name"],
                **{f"{name} IoU (%)": percent(evaluation["class_iou"].get(name)) for name in class_names},
            }
        )

    if len(tasks) != 1:
        raise ValueError("Per-class comparison requires configurations using the same label task.")
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_csv = output_directory / f"{args.summary_stem}.csv"
    classes_csv = output_directory / f"{args.classes_stem}.csv"
    for path, content in [(summary_csv, rows), (classes_csv, class_rows)]:
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(content[0].keys()))
            writer.writeheader()
            writer.writerows(content)

    markdown_path = output_directory / f"{args.summary_stem}.md"
    with markdown_path.open("w", encoding="utf-8") as output_file:
        for title, content in [(args.title, rows), ("Per-Class IoU", class_rows)]:
            headers = list(content[0].keys())
            output_file.write(f"## {title}\n\n")
            output_file.write("| " + " | ".join(headers) + " |\n")
            output_file.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in content:
                output_file.write("| " + " | ".join(str(row[header]) for header in headers) + " |\n")
            output_file.write("\n")

    print(f"Comparison table saved to: {markdown_path}")


if __name__ == "__main__":
    main()
