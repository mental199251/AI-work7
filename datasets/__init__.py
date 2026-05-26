from .transforms import build_eval_transform, build_train_transform
from .voc_five_class import (
    CLASS_COLORS,
    CLASS_NAMES,
    FIVE_CLASS_COLORS,
    FIVE_CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    TASK_SPECS,
    VOC21_CLASS_COLORS,
    VOC21_CLASS_NAMES,
    VOCFiveClassSegmentation,
    VOCSegmentationTask,
    get_task_spec,
    remap_voc_mask,
)

__all__ = [
    "CLASS_COLORS",
    "CLASS_NAMES",
    "FIVE_CLASS_COLORS",
    "FIVE_CLASS_NAMES",
    "IGNORE_INDEX",
    "NUM_CLASSES",
    "TASK_SPECS",
    "VOC21_CLASS_COLORS",
    "VOC21_CLASS_NAMES",
    "VOCFiveClassSegmentation",
    "VOCSegmentationTask",
    "build_eval_transform",
    "build_train_transform",
    "get_task_spec",
    "remap_voc_mask",
]
