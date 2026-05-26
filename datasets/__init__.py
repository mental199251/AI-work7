from .transforms import build_eval_transform, build_train_transform
from .voc_five_class import (
    CLASS_COLORS,
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    VOCFiveClassSegmentation,
)

__all__ = [
    "CLASS_COLORS",
    "CLASS_NAMES",
    "IGNORE_INDEX",
    "NUM_CLASSES",
    "VOCFiveClassSegmentation",
    "build_eval_transform",
    "build_train_transform",
]
