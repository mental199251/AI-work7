import numpy as np
from PIL import Image

from datasets import IGNORE_INDEX, get_task_spec, remap_voc_mask


def test_five_class_mapping_ignores_unselected_voc_classes() -> None:
    raw_mask = Image.fromarray(np.array([[0, 15, 7, 8, 12, 1, 255]], dtype=np.uint8))

    remapped = np.asarray(remap_voc_mask(raw_mask, "five_class"))

    assert remapped.tolist() == [[0, 1, 2, 3, 4, IGNORE_INDEX, IGNORE_INDEX]]


def test_voc21_mapping_preserves_original_class_ids_and_ignore_boundary() -> None:
    raw = np.array([[0, 1, 7, 12, 15, 20, 255]], dtype=np.uint8)

    remapped = np.asarray(remap_voc_mask(Image.fromarray(raw), "voc21"))

    assert np.array_equal(remapped, raw)
    assert get_task_spec("voc21").num_classes == 21
