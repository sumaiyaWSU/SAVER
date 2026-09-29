from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

WEATHER_KEYS = ["Sunny", "Rainy", "Snow", "Foggy", "Sand"]
SCENE_KEYS = ["Highway", "Downtown"]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Frame:
    weather: str
    scene: str
    sequence_id: str
    frame_index: int
    image_path: str
    is_new_sequence: bool
    ground_truth: List[Dict[str, Any]] = field(default_factory=list)


def _load_annotations(path: str) -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    anns = data.get("annotations", {})
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rel_path, boxes in anns.items():
        cleaned = []
        for b in boxes:
            cleaned.append({
                "category": str(b["category"]).strip().lower(),
                "box": [float(v) for v in b["bbox"]],
            })
        out[rel_path.replace("\\", "/")] = cleaned
    return out


def _load_bucket(images_dir: str, annotations: Dict[str, List[Dict[str, Any]]],
                  weather: str, scene: str) -> List[Frame]:
    if not os.path.isdir(images_dir):
        return []

    sequences: Dict[str, List[str]] = {}
    for entry in sorted(os.listdir(images_dir)):
        full = os.path.join(images_dir, entry)
        if os.path.isdir(full):
            frame_files = sorted(
                f for f in os.listdir(full)
                if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
            )
            if frame_files:
                if entry in sequences:
                    raise ValueError(
                        f"Naming collision in {images_dir!r}: sequence folder "
                        f"'{entry}/' collides with a standalone image "
                        f"'{entry}.<ext>' -- rename one of them (each sequence "
                        f"ID / standalone image stem must be unique within a "
                        f"bucket's images/ folder)."
                    )
                sequences[entry] = [os.path.join(entry, f) for f in frame_files]
        elif os.path.splitext(entry)[1].lower() in _IMAGE_EXTS:
            stem = os.path.splitext(entry)[0]
            if stem in sequences:
                raise ValueError(
                    f"Naming collision in {images_dir!r}: standalone image "
                    f"'{entry}' collides with sequence folder '{stem}/' -- "
                    f"rename one of them (each sequence ID / standalone image "
                    f"stem must be unique within a bucket's images/ folder)."
                )
            sequences[stem] = [entry]

    frames: List[Frame] = []
    for sequence_id in sorted(sequences.keys()):
        rel_paths = sequences[sequence_id]
        for idx, rel_path in enumerate(rel_paths):
            rel_path = rel_path.replace("\\", "/")
            frames.append(Frame(
                weather=weather,
                scene=scene,
                sequence_id=sequence_id,
                frame_index=idx,
                image_path=os.path.join(images_dir, *rel_path.split("/")),
                is_new_sequence=(idx == 0),
                ground_truth=annotations.get(rel_path, []),
            ))
    return frames


def load_dataset(root: str) -> Dict[Tuple[str, str], List[Frame]]:
    """Returns {(weather_key, scene_key): [Frame, ...]} for every bucket
    that actually exists under `root` (missing weather/scene folders are
    silently skipped, so a partial dataset -- e.g. only Sunny/Foggy so
    far -- still loads and evaluates whatever is present)."""
    buckets: Dict[Tuple[str, str], List[Frame]] = {}
    for weather in WEATHER_KEYS:
        for scene in SCENE_KEYS:
            bucket_dir = os.path.join(root, weather, scene)
            if not os.path.isdir(bucket_dir):
                continue
            images_dir = os.path.join(bucket_dir, "images")
            annotations = _load_annotations(os.path.join(bucket_dir, "annotations.json"))
            frames = _load_bucket(images_dir, annotations, weather, scene)
            if frames:
                buckets[(weather, scene)] = frames
    return buckets


def summarize(buckets: Dict[Tuple[str, str], List[Frame]]) -> str:
    lines = []
    total_frames = 0
    total_sequences = 0
    for (weather, scene), frames in sorted(buckets.items()):
        n_seq = len({f.sequence_id for f in frames})
        n_gt = sum(len(f.ground_truth) for f in frames)
        lines.append(f"  {weather:8s}/{scene:9s}: {len(frames):5d} frames, "
                      f"{n_seq:4d} sequences, {n_gt:6d} ground-truth boxes")
        total_frames += len(frames)
        total_sequences += n_seq
    lines.append(f"  TOTAL: {total_frames} frames across {total_sequences} sequences, "
                 f"{len(buckets)} weather/scene buckets")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a Table VIII evaluation dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset root directory")
    args = parser.parse_args()

    buckets = load_dataset(args.dataset)
    if not buckets:
        print(f"No weather/scene buckets found under {args.dataset}. "
              f"Expected <root>/<{'|'.join(WEATHER_KEYS)}>/<{'|'.join(SCENE_KEYS)}>/images/")
    else:
        print(summarize(buckets))