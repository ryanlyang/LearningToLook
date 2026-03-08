"""Compatibility loader for dataset-specific CLIP text configurations."""

import importlib
import os
from typing import Dict, List, Tuple

from .clip_texts import DATASET_MODULES


_DATASET_ALIASES: Dict[str, str] = {
    "waterbirds": "waterbirds",
    "waterbird": "waterbirds",
    "bird": "waterbirds",
    "redmeat": "redmeat",
    "red_meat": "redmeat",
    "meat": "redmeat",
    "decoymnist": "decoymnist",
    "decoy_mnist": "decoymnist",
    "mnist": "decoymnist",
    "digit": "decoymnist",
}

_CLASS_TO_DATASET: Dict[str, str] = {
    "bird": "waterbirds",
    "meat": "redmeat",
    "digit": "decoymnist",
}


def _normalize_token(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_dataset_name() -> str:
    requested = os.environ.get("CLIP_TEXT_DATASET", "")
    if requested:
        resolved = _DATASET_ALIASES.get(_normalize_token(requested), None)
        if resolved in DATASET_MODULES:
            return resolved

    version = os.environ.get("CLIP_TEXT_VERSION", "")
    if version:
        class_key = version.replace("_", " ").strip().lower()
        resolved = _CLASS_TO_DATASET.get(class_key, None)
        if resolved in DATASET_MODULES:
            return resolved

    default_name = _normalize_token(os.environ.get("CLIP_TEXT_DEFAULT_DATASET", "redmeat"))
    resolved = _DATASET_ALIASES.get(default_name, None)
    if resolved in DATASET_MODULES:
        return resolved
    return "redmeat"


def _load_dataset_spec(dataset_name: str) -> Tuple[List[str], List[str], List[str]]:
    module = importlib.import_module(DATASET_MODULES[dataset_name])

    bg = list(getattr(module, "BACKGROUND_CATEGORY"))
    all_classes = list(getattr(module, "_all_class_names"))
    all_new_classes = list(getattr(module, "_all_new_class_names"))

    if len(all_classes) != len(all_new_classes):
        raise RuntimeError(
            f"Invalid CLIP text config '{dataset_name}': "
            f"_all_class_names ({len(all_classes)}) and _all_new_class_names "
            f"({len(all_new_classes)}) must have the same length."
        )
    return bg, all_classes, all_new_classes


ACTIVE_CLIP_TEXT_DATASET = _resolve_dataset_name()
BACKGROUND_CATEGORY, _all_class_names, _all_new_class_names = _load_dataset_spec(
    ACTIVE_CLIP_TEXT_DATASET
)

_version = os.environ.get("CLIP_TEXT_VERSION", None)
if _version:
    _version = _version.replace("_", " ").strip()

if _version and _version in _all_class_names:
    idx = _all_class_names.index(_version)
    class_names = [_all_class_names[idx]]
    new_class_names = [_all_new_class_names[idx]]
else:
    class_names = _all_class_names
    new_class_names = _all_new_class_names


class_names_coco = [
    "person", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

new_class_names_coco = [
    "person with clothes,people,human", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird avian",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack,bag",
    "umbrella,parasol", "handbag,purse", "necktie", "suitcase", "frisbee",
    "skis", "sknowboard", "sports ball", "kite", "baseball bat",
    "glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "dessertspoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair seat", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor screen", "laptop", "mouse",
    "remote control", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hairdrier,blowdrier", "toothbrush",
]

BACKGROUND_CATEGORY_COCO = [
    "ground", "land", "grass", "tree", "building", "wall", "sky", "lake", "water",
    "river", "sea", "railway", "railroad", "helmet", "cloud", "house", "mountain",
    "ocean", "road", "rock", "street", "valley", "bridge",
]
