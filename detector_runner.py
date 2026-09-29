from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image, ImageDraw


CHECKPOINT_OVERRIDES_PATH = os.environ.get(
    "SAVER_DETECTOR_CHECKPOINTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_checkpoint_overrides.json"),
)


def _load_checkpoint_overrides(path: str) -> Dict[str, str]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


_CHECKPOINT_OVERRIDES = _load_checkpoint_overrides(CHECKPOINT_OVERRIDES_PATH)


@dataclass
class DetectionRunResult:
    model: str
    latency_ms: float
    confidence: float
    num_detections: int
    detections: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = ""
    device: str = ""


class DetectorUnavailableError(RuntimeError):
    """Raised when a registered detector cannot be loaded/run in this environment."""


_MODEL_CACHE: Dict[str, Any] = {}
_CUSTOM_LOADERS: Dict[str, Callable[[], Any]] = {}

DEFAULT_SCORE_THRESHOLD = 0.5


def register_custom_loader(model_name: str, loader: Callable[[], Any], runner: Callable[[Any, Image.Image, float], List[Dict[str, Any]]]) -> None:
    """Extension point for detectors with no built-in backend (e.g. Sparse-RCNN, DINO).

    `loader` takes no args and returns a ready-to-use model object (cached).
    `runner` takes (model, PIL image, score_threshold) and returns a list of
    {"label": str, "score": float, "box": [x1, y1, x2, y2]} dicts.
    """
    _CUSTOM_LOADERS[model_name] = (loader, runner)


def _require_import(module_name: str, pip_name: Optional[str] = None):
    """Imports `module_name`, raising a clear DetectorUnavailableError with
    an actionable `pip install` command instead of a bare ModuleNotFoundError
    if it's missing -- this is what surfaces when a backend's package just
    isn't installed in this environment yet (e.g. `pip install ultralytics`
    was skipped)."""
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise DetectorUnavailableError(
            f"'{module_name}' is not installed in this Python environment "
            f"({exc}). Fix: pip install {pip_name or module_name}"
        ) from exc


def _load_torchvision(model_name: str):
    _require_import("torchvision")
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
    )

    override_path = _CHECKPOINT_OVERRIDES.get(model_name)

    if model_name == "Faster-RCNN":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=None if override_path else weights, box_score_thresh=0.05)
    elif model_name == "Mask-RCNN":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=None if override_path else weights, box_score_thresh=0.05)
    else:
        raise DetectorUnavailableError(f"No torchvision loader for '{model_name}'")

    if override_path:
        state_dict = torch.load(override_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return {"model": model, "weights": weights, "device": device, "labels": weights.meta["categories"]}


def _run_torchvision(handle, image: Image.Image, score_threshold: float) -> List[Dict[str, Any]]:
    transform = handle["weights"].transforms()
    x = transform(image).to(handle["device"]).unsqueeze(0)
    with torch.no_grad():
        output = handle["model"](x)[0]

    labels = handle["labels"]
    dets = []
    for box, score, label_idx in zip(output["boxes"].tolist(), output["scores"].tolist(), output["labels"].tolist()):
        if score < score_threshold:
            continue
        dets.append({
            "label": labels[label_idx] if 0 <= label_idx < len(labels) else str(label_idx),
            "score": float(score),
            "box": [float(v) for v in box],
        })
    return dets


_HF_CHECKPOINTS = {
    "DETR": "facebook/detr-resnet-50",
    "Deformable DETR (R50)": "SenseTime/deformable-detr",
    "RT-DETR-L": "PekingU/rtdetr_r50vd",
}


def _load_transformers(model_name: str):
    _require_import("transformers")
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    checkpoint = _CHECKPOINT_OVERRIDES.get(model_name) or _HF_CHECKPOINTS.get(model_name)
    if checkpoint is None:
        raise DetectorUnavailableError(f"No transformers checkpoint mapped for '{model_name}'")

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return {"model": model, "processor": processor, "device": device}


def _run_transformers(handle, image: Image.Image, score_threshold: float) -> List[Dict[str, Any]]:
    processor = handle["processor"]
    model = handle["model"]
    device = handle["device"]

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=score_threshold, target_sizes=target_sizes
    )[0]

    id2label = model.config.id2label
    dets = []
    for score, label_idx, box in zip(results["scores"].tolist(), results["labels"].tolist(), results["boxes"].tolist()):
        dets.append({
            "label": id2label.get(label_idx, str(label_idx)),
            "score": float(score),
            "box": [float(v) for v in box],
        })
    return dets


_ULTRALYTICS_WEIGHTS = {
    "YOLOv11(Large)": "yolo11l.pt",
}


def _load_ultralytics(model_name: str):
    _require_import("ultralytics")
    from ultralytics import YOLO

    weights = _CHECKPOINT_OVERRIDES.get(model_name) or _ULTRALYTICS_WEIGHTS.get(model_name)
    if weights is None:
        raise DetectorUnavailableError(f"No ultralytics weights mapped for '{model_name}'")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)
    model.to(device)
    return {"model": model, "device": device}


def _run_ultralytics(handle, image: Image.Image, score_threshold: float) -> List[Dict[str, Any]]:
    results = handle["model"].predict(image, conf=score_threshold, device=handle["device"], verbose=False)
    if not results:
        return []
    result = results[0]
    names = result.names
    dets = []
    boxes = result.boxes
    if boxes is None:
        return dets
    for box, score, cls_idx in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
        dets.append({
            "label": names.get(int(cls_idx), str(int(cls_idx))) if isinstance(names, dict) else str(names[int(cls_idx)]),
            "score": float(score),
            "box": [float(v) for v in box],
        })
    return dets


SPARSE_RCNN_REPO = os.environ.get("SPARSE_RCNN_REPO")
SPARSE_RCNN_WEIGHTS = os.environ.get("SPARSE_RCNN_WEIGHTS")
SPARSE_RCNN_CONFIG = os.environ.get(
    "SPARSE_RCNN_CONFIG", "projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml"
)


def _load_sparsercnn(model_name: str):
    if not SPARSE_RCNN_REPO:
        raise DetectorUnavailableError(
            "Sparse-RCNN requires a local clone of the official repo "
            "(https://github.com/PeizeSun/SparseR-CNN) plus detectron2. Clone "
            "the repo, install detectron2, download a checkpoint from the "
            "repo's README model zoo, then set SPARSE_RCNN_REPO (path to the "
            "clone) and SPARSE_RCNN_WEIGHTS (path to the checkpoint) before "
            "running -- see the comment above _load_sparsercnn for details."
        )
    if not SPARSE_RCNN_WEIGHTS:
        raise DetectorUnavailableError(
            "SPARSE_RCNN_REPO is set but SPARSE_RCNN_WEIGHTS is not. Download "
            "a checkpoint from the SparseR-CNN repo's README (model zoo) and "
            "point SPARSE_RCNN_WEIGHTS at that file."
        )

    projects_dir = os.path.join(SPARSE_RCNN_REPO, "projects", "SparseRCNN")
    if projects_dir not in sys.path:
        sys.path.insert(0, projects_dir)

    try:
        from detectron2.config import get_cfg
        from detectron2.data import MetadataCatalog
        from detectron2.engine import DefaultPredictor
        from sparsercnn import add_sparsercnn_config
    except ImportError as exc:
        raise DetectorUnavailableError(
            f"Could not import detectron2 / the SparseR-CNN repo's 'sparsercnn' "
            f"package from '{projects_dir}': {exc}. Confirm detectron2 is "
            f"installed and SPARSE_RCNN_REPO points at a full clone of "
            f"https://github.com/PeizeSun/SparseR-CNN (with its projects/SparseRCNN "
            f"directory intact)."
        ) from exc

    config_path = SPARSE_RCNN_CONFIG
    if not os.path.isabs(config_path):
        config_path = os.path.join(SPARSE_RCNN_REPO, config_path)

    cfg = get_cfg()
    add_sparsercnn_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = SPARSE_RCNN_WEIGHTS
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    try:
        dataset_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "coco_2017_val"
        thing_classes = MetadataCatalog.get(dataset_name).thing_classes
    except Exception:
        thing_classes = None

    return {"predictor": predictor, "device": cfg.MODEL.DEVICE, "thing_classes": thing_classes}


def _run_sparsercnn(handle, image: Image.Image, score_threshold: float) -> List[Dict[str, Any]]:
    import numpy as np

    predictor = handle["predictor"]
    image_bgr = np.array(image)[:, :, ::-1]
    outputs = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")

    thing_classes = handle.get("thing_classes")
    boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
    scores = instances.scores.tolist() if instances.has("scores") else []
    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []

    dets = []
    for box, score, cls_idx in zip(boxes, scores, classes):
        if score < score_threshold:
            continue
        label = (
            thing_classes[cls_idx]
            if thing_classes and 0 <= cls_idx < len(thing_classes)
            else str(cls_idx)
        )
        dets.append({"label": label, "score": float(score), "box": [float(v) for v in box]})
    return dets


DINO_REPO = os.environ.get("DINO_REPO")
DINO_WEIGHTS = os.environ.get("DINO_WEIGHTS")
DINO_CONFIG = os.environ.get("DINO_CONFIG", "config/DINO/DINO_4scale.py")


def _load_dino(model_name: str):
    """DINO (DETR with Improved deNoising anchOr boxes) has no torchvision/
    HF-transformers/ultralytics backend -- as of this writing `transformers`
    ships DINOv2/DINOv3 (a different, self-supervised ViT backbone) and
    Grounding DINO (a different, open-vocabulary detector), but not this
    plain closed-set object detector. The official repo is
    https://github.com/IDEA-Research/DINO (NOT facebookresearch/dino, 
    Please load the official repo clone from IDEA-Research).
    """
    if not DINO_REPO:
        raise DetectorUnavailableError(
            "DINO requires a local clone of the official repo "
            "(https://github.com/IDEA-Research/DINO) with its custom "
            "deformable-attention CUDA extension compiled. Clone the repo, "
            "`pip install -r requirements.txt`, run `cd models/dino/ops && "
            "python setup.py build install`, download a checkpoint from the "
            "repo's README model zoo, then set DINO_REPO (path to the clone) "
            "and DINO_WEIGHTS (path to the checkpoint) before running -- see "
            "the comment above _load_dino for details."
        )
    if not DINO_WEIGHTS:
        raise DetectorUnavailableError(
            "DINO_REPO is set but DINO_WEIGHTS is not. Download a checkpoint "
            "from the DINO repo's README (model zoo) and point DINO_WEIGHTS "
            "at that file."
        )

    if DINO_REPO not in sys.path:
        sys.path.insert(0, DINO_REPO)

    try:
        from main import build_model_main
        from util.slconfig import SLConfig
        import datasets.transforms as DT
    except ImportError as exc:
        raise DetectorUnavailableError(
            f"Could not import DINO's own modules (main.build_model_main / "
            f"util.slconfig.SLConfig / datasets.transforms) from "
            f"'{DINO_REPO}': {exc}. Confirm DINO_REPO points at a full clone "
            f"of https://github.com/IDEA-Research/DINO with its requirements "
            f"installed and its deformable-attention CUDA op compiled."
        ) from exc

    config_path = DINO_CONFIG
    if not os.path.isabs(config_path):
        config_path = os.path.join(DINO_REPO, config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = SLConfig.fromfile(config_path)
    args.device = device
    model, _, postprocessors = build_model_main(args)

    checkpoint = torch.load(DINO_WEIGHTS, map_location="cpu")
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    model.to(device)

    transform = DT.Compose([
        DT.RandomResize([800], max_size=1333),
        DT.ToTensor(),
        DT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    id2name = None
    for candidate in ("util/coco_id2name.json", "util/coco_id2name_v2.json"):
        candidate_path = os.path.join(DINO_REPO, candidate)
        if os.path.exists(candidate_path):
            with open(candidate_path, "r") as f:
                raw = json.load(f)
            id2name = {int(k): v for k, v in raw.items()}
            break

    return {
        "model": model, "postprocessors": postprocessors, "transform": transform,
        "device": device, "id2name": id2name,
    }


def _run_dino(handle, image: Image.Image, score_threshold: float) -> List[Dict[str, Any]]:
    model = handle["model"]
    device = handle["device"]
    id2name = handle.get("id2name")

    image_transformed, _ = handle["transform"](image, None)
    with torch.no_grad():
        output = model(image_transformed[None].to(device))

    orig_size = torch.tensor([[image.size[1], image.size[0]]], device=device)  # (h, w)
    result = handle["postprocessors"]["bbox"](output, orig_size)[0]

    scores = result["scores"].tolist()
    labels = result["labels"].tolist()
    boxes = result["boxes"].tolist()

    dets = []
    for score, label_idx, box in zip(scores, labels, boxes):
        if score < score_threshold:
            continue
        dets.append({
            "label": id2name[label_idx] if id2name and label_idx in id2name else str(label_idx),
            "score": float(score),
            "box": [float(v) for v in box],
        })
    return dets


_BACKENDS: Dict[str, Any] = {
    "Faster-RCNN": ("torchvision", _load_torchvision, _run_torchvision),
    "Mask-RCNN": ("torchvision", _load_torchvision, _run_torchvision),
    "DETR": ("transformers", _load_transformers, _run_transformers),
    "Deformable DETR (R50)": ("transformers", _load_transformers, _run_transformers),
    "RT-DETR-L": ("transformers", _load_transformers, _run_transformers),
    "YOLOv11(Large)": ("ultralytics", _load_ultralytics, _run_ultralytics),
    "Sparse-RCNN": ("detectron2", _load_sparsercnn, _run_sparsercnn),
    "DINO": ("dino-repo", _load_dino, _run_dino),
}


def is_supported(model_name: str) -> bool:
    return model_name in _BACKENDS or model_name in _CUSTOM_LOADERS


def supported_models() -> List[str]:
    return sorted(set(_BACKENDS) | set(_CUSTOM_LOADERS))


_WARMED_UP: set = set()


def _get_handle(model_name: str, loader: Callable[[str], Any]):
    is_new = model_name not in _MODEL_CACHE
    if is_new:
        _MODEL_CACHE[model_name] = loader(model_name)
    return _MODEL_CACHE[model_name], is_new


def run_detector(
    model_name: str,
    image_path: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> DetectionRunResult:
    """Load (if needed) and run `model_name` from the Winner Model Database
    on `image_path`, returning its measured confidence and inference latency.

    Raises DetectorUnavailableError if the model has no usable backend/weights
    in this environment (missing package, no network to fetch weights, no
    custom loader registered, etc.). note-Please run the actual detection model for accurate results.
    """
    image = Image.open(image_path).convert("RGB")

    if model_name in _CUSTOM_LOADERS:
        loader, runner = _CUSTOM_LOADERS[model_name]
        backend_name = "custom"
    elif model_name in _BACKENDS:
        backend_name, loader_fn, runner = _BACKENDS[model_name]
        loader = lambda: loader_fn(model_name)
    else:
        raise DetectorUnavailableError(
            f"'{model_name}' is not registered with a runnable backend. "
            f"Supported out of the box: {supported_models()}. "
            f"Use detector_runner.register_custom_loader(...) to add it."
        )

    try:
        handle, is_new_load = _get_handle(model_name, lambda _name: loader())
    except DetectorUnavailableError:
        raise
    except Exception as exc:
        raise DetectorUnavailableError(f"Could not load '{model_name}': {exc}") from exc

    device = handle.get("device", "unknown") if isinstance(handle, dict) else "unknown"
    if is_new_load and device == "cpu":
        print(f"[detector_runner] WARNING: '{model_name}' is running on CPU (no CUDA "
              f"GPU available in this process). Latency will be far higher than the "
              f"paper's GPU-profiled reference numbers -- check your torch/CUDA install.")

    if is_new_load and model_name not in _WARMED_UP:
        try:
            runner(handle, image, score_threshold)
        except Exception:
            pass
        _WARMED_UP.add(model_name)

    try:
        start = time.time()
        detections = runner(handle, image, score_threshold)
        latency_ms = (time.time() - start) * 1000.0
    except Exception as exc:
        raise DetectorUnavailableError(f"Inference failed for '{model_name}': {exc}") from exc

    confidence = sum(d["score"] for d in detections) / len(detections) if detections else 0.0

    return DetectionRunResult(
        model=model_name,
        latency_ms=latency_ms,
        confidence=confidence,
        num_detections=len(detections),
        detections=detections,
        backend=backend_name,
        device=device,
    )


def run_with_fallback(
    candidate_models: List[str],
    image_path: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> Optional[DetectionRunResult]:
    """Try candidates in order (RL/CoT choice first) and return the first one
    that actually runs. Returns None if none of the candidates are runnable
    in this environment (e.g. no network to fetch weights) so the user can
    degrade gracefully instead of CRASHING the whole pipeline.
    """
    last_error = None
    for name in candidate_models:
        try:
            return run_detector(name, image_path, score_threshold=score_threshold)
        except DetectorUnavailableError as exc:
            last_error = exc
            print(f"[detector_runner] '{name}' unavailable, trying next candidate: {exc}")
            continue
    if last_error is not None:
        print(f"[detector_runner] no candidate in {candidate_models} could be executed; last error: {last_error}")
    return None


_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


def _color_for_label(label: str) -> tuple:
    idx = zlib.crc32(label.encode("utf-8")) % len(_PALETTE)
    return _PALETTE[idx]


def save_annotated_image(
    image_path: str,
    detections: List[Dict[str, Any]],
    out_path: str,
    title: Optional[str] = None,
) -> str:
    """Draws the real detection boxes/labels/scores from a run_detector()
    result onto a copy of the input frame and saves it to out_path, so the
    actual output of the executed detector -- not just its JSON summary --
    can be inspected. Each distinct class label gets its own deterministic
    color (see _PALETTE/_color_for_label) rather than one color for every
    box. Returns out_path."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = _color_for_label(d["label"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f'{d["label"]} {d["score"]:.2f}'
        text_y = max(0, y1 - 12)
        draw.rectangle([x1, text_y, x1 + 7 * len(label_text), text_y + 11], fill=color)
        draw.text((x1 + 2, text_y), label_text, fill=(255, 255, 255))

    if title:
        draw.rectangle([0, 0, 8 * len(title) + 8, 16], fill=(0, 0, 0))
        draw.text((4, 2), title, fill=(0, 255, 0))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    image.save(out_path)
    return out_path


def describe_environment() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": None,
        "packages": {},
        "models": {},
    }
    if report["cuda_available"]:
        try:
            report["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            report["cuda_device"] = "unknown"

    for pkg in ("torchvision", "transformers", "ultralytics", "detectron2"):
        try:
            mod = __import__(pkg)
            report["packages"][pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            report["packages"][pkg] = None

    for model_name in sorted(set(_BACKENDS) | set(_CUSTOM_LOADERS)):
        if model_name in _BACKENDS:
            backend_name = _BACKENDS[model_name][0]
        elif model_name in _CUSTOM_LOADERS:
            backend_name = "custom"
        else:
            backend_name = None
        info: Dict[str, Any] = {
            "backend": backend_name,
            "supported": is_supported(model_name),
            "checkpoint_override": _CHECKPOINT_OVERRIDES.get(model_name),
        }
        if model_name == "Sparse-RCNN":
            info["SPARSE_RCNN_REPO_set"] = bool(SPARSE_RCNN_REPO)
            info["SPARSE_RCNN_WEIGHTS_set"] = bool(SPARSE_RCNN_WEIGHTS)
        if model_name == "DINO":
            info["DINO_REPO_set"] = bool(DINO_REPO)
            info["DINO_WEIGHTS_set"] = bool(DINO_WEIGHTS)
        report["models"][model_name] = info
    return report


def print_environment_report() -> None:
    report = describe_environment()
    print("=== detector_runner environment report ===")
    gpu = f"  ({report['cuda_device']})" if report["cuda_device"] else ""
    print(f"torch {report['torch']}  |  CUDA available: {report['cuda_available']}{gpu}")
    print("\npackages:")
    for pkg, version in report["packages"].items():
        print(f"  {pkg:14s}: {'MISSING' if version is None else version}")
    print(f"\ncheckpoint overrides file: {CHECKPOINT_OVERRIDES_PATH} "
          f"({'found' if _CHECKPOINT_OVERRIDES else 'not found / empty'})")
    print("\nmodels (Winner Model Database):")
    for name, info in report["models"].items():
        line = f"  {name:24s} backend={str(info['backend']):12s} supported={info['supported']}"
        if info.get("checkpoint_override"):
            line += f"  override={info['checkpoint_override']}"
        if "SPARSE_RCNN_REPO_set" in info:
            line += (f"  SPARSE_RCNN_REPO set={info['SPARSE_RCNN_REPO_set']}"
                      f"  SPARSE_RCNN_WEIGHTS set={info['SPARSE_RCNN_WEIGHTS_set']}")
        if "DINO_REPO_set" in info:
            line += (f"  DINO_REPO set={info['DINO_REPO_set']}"
                      f"  DINO_WEIGHTS set={info['DINO_WEIGHTS_set']}")
        print(line)
    unsupported = [n for n, i in report["models"].items() if not i["supported"]]
    if unsupported:
        print(f"\nNo backend registered for: {unsupported}. Use "
              f"register_custom_loader(...) to add one -- see the module "
              f"docstring for the extension point's signature.")


def preload_all(only: Optional[List[str]] = None, score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> Dict[str, str]:
    """Triggers every registered backend's loader once against a throwaway
    image, forcing weight download/caching now instead of during a real
    pipeline run. Returns {model_name: "OK" | "UNAVAILABLE: <reason>"}."""
    tmp_path = os.path.join(tempfile.gettempdir(), "_detector_runner_preload.jpg")
    Image.new("RGB", (256, 256), (128, 128, 128)).save(tmp_path)

    names = only or supported_models()
    results: Dict[str, str] = {}
    for name in names:
        print(f"\n[preload] {name} ...")
        try:
            result = run_detector(name, tmp_path, score_threshold=score_threshold)
            results[name] = "OK"
            print(f"[preload] {name}: OK (backend={result.backend}, device={result.device})")
        except DetectorUnavailableError as exc:
            results[name] = f"UNAVAILABLE: {exc}"
            print(f"[preload] {name}: UNAVAILABLE - {exc}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect the detector_runner environment and optionally preload (download/cache) model weights."
    )
    parser.add_argument("--preload", action="store_true",
                         help="Download/cache weights for every registered backend now.")
    parser.add_argument("--only", default=None,
                         help="Comma-separated subset of model names to preload (default: all supported).")
    args = parser.parse_args()

    print_environment_report()

    if args.preload:
        only = [m.strip() for m in args.only.split(",")] if args.only else None
        print("\n=== Preloading ===")
        results = preload_all(only=only)
        print("\n=== Preload summary ===")
        for name, status in results.items():
            print(f"  {name}: {status}")