import argparse

from datasets import build_eval_transform
from datasets.voc_five_class import VOCFiveClassSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and inspect the PASCAL VOC 2012 five-class subset.")
    parser.add_argument("--root", default="./data")
    parser.add_argument("--year", default="2012")
    parser.add_argument("--input-size", nargs=2, type=int, default=[320, 320])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-filter-foreground", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = build_eval_transform({"input_size": args.input_size})
    for split in ["train", "val"]:
        dataset = VOCFiveClassSegmentation(
            root=args.root,
            year=args.year,
            image_set=split,
            transform=transform,
            download=args.download,
            filter_foreground=not args.no_filter_foreground,
        )
        print(f"{split}: {len(dataset)} selected images")


if __name__ == "__main__":
    main()
