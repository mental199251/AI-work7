import argparse

from datasets import VOCSegmentationTask, build_eval_transform, get_task_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and inspect the PASCAL VOC 2012 five-class subset.")
    parser.add_argument("--root", default="./data")
    parser.add_argument("--year", default="2012")
    parser.add_argument("--input-size", nargs=2, type=int, default=[320, 320])
    parser.add_argument("--task", choices=["five_class", "voc21"], default="five_class")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--filter-foreground", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = build_eval_transform({"input_size": args.input_size})
    filter_foreground = args.task == "five_class" if args.filter_foreground is None else args.filter_foreground
    task_spec = get_task_spec(args.task)
    print(f"Task: {args.task} ({task_spec.num_classes} classes), filter_foreground={filter_foreground}")
    for split in ["train", "val"]:
        dataset = VOCSegmentationTask(
            root=args.root,
            year=args.year,
            image_set=split,
            transform=transform,
            download=args.download,
            filter_foreground=filter_foreground,
            task=args.task,
        )
        print(f"{split}: {len(dataset)} selected images")


if __name__ == "__main__":
    main()
