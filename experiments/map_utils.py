from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0
    return inter_area / union


def _all_point_interpolated_ap(recalls: List[float], precisions: List[float]) -> float:
    """COCO/VOC-2012-style all-point interpolated average precision: the
    area under the precision-recall curve after replacing each precision
    value with the max precision at any recall >= that point (monotone
    envelope), integrated over recall via the trapezoid rule at the
    (already sorted) recall breakpoints."""
    if not recalls:
        return 0.0

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def average_precision_for_category(
    detections: List[Dict[str, Any]],
    ground_truths: Dict[Any, List[Dict[str, Any]]],
    num_gt: int,
    iou_threshold: float = 0.5,
) -> float:
    """detections: [{"image_id":.., "score":.., "box":[x1,y1,x2,y2]}, ...]
    for ONE category, already filtered to that category.
    ground_truths: {image_id: [{"box":[..], "matched": False}, ...]} for
    the SAME category -- mutated in place to track matches, so pass a
    fresh copy per call.
    num_gt: total number of ground-truth boxes for this category (over
    all images, including images with zero detections).
    Returns 0.0 if num_gt == 0 (nothing to detect: undefined, not a bug).
    """
    if num_gt == 0:
        return 0.0
    if not detections:
        return 0.0

    dets_sorted = sorted(detections, key=lambda d: d["score"], reverse=True)

    tp = [0.0] * len(dets_sorted)
    fp = [0.0] * len(dets_sorted)

    for i, det in enumerate(dets_sorted):
        candidates = ground_truths.get(det["image_id"], [])
        best_iou = 0.0
        best_gt = None
        for gt in candidates:
            if gt["matched"]:
                continue
            cur_iou = iou(det["box"], gt["box"])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt = gt
        if best_gt is not None and best_iou >= iou_threshold:
            best_gt["matched"] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    cum_tp = 0.0
    cum_fp = 0.0
    recalls: List[float] = []
    precisions: List[float] = []
    for i in range(len(dets_sorted)):
        cum_tp += tp[i]
        cum_fp += fp[i]
        recalls.append(cum_tp / num_gt)
        precisions.append(cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0)

    return _all_point_interpolated_ap(recalls, precisions)


def compute_map(
    detections: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    categories: Optional[List[str]] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """detections: [{"image_id":.., "category":.., "score":.., "box":[..]}, ...]
    ground_truths: [{"image_id":.., "category":.., "box":[..]}, ...]
    categories: category names to average over; defaults to every category
    that appears in `ground_truths`.

    Returns {"mAP": float, "per_category": {cat: AP}, "num_categories": int}.
    """
    gt_by_category: Dict[str, Dict[Any, List[Dict[str, Any]]]] = {}
    num_gt_by_category: Dict[str, int] = {}
    for gt in ground_truths:
        cat = gt["category"]
        gt_by_category.setdefault(cat, {}).setdefault(gt["image_id"], []).append(
            {"box": gt["box"], "matched": False}
        )
        num_gt_by_category[cat] = num_gt_by_category.get(cat, 0) + 1

    if categories is None:
        categories = sorted(num_gt_by_category.keys())

    dets_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        dets_by_category.setdefault(det["category"], []).append(det)

    per_category: Dict[str, float] = {}
    for cat in categories:
        ap = average_precision_for_category(
            dets_by_category.get(cat, []),
            gt_by_category.get(cat, {}),
            num_gt_by_category.get(cat, 0),
            iou_threshold=iou_threshold,
        )
        per_category[cat] = ap

    mAP = sum(per_category.values()) / len(per_category) if per_category else 0.0
    return {"mAP": mAP, "per_category": per_category, "num_categories": len(per_category)}


def _self_test() -> None:
    dets = [{"image_id": 1, "category": "car", "score": 0.9, "box": [0, 0, 10, 10]}]
    gts = [{"image_id": 1, "category": "car", "box": [0, 0, 10, 10]}]
    result = compute_map(dets, gts)
    assert abs(result["mAP"] - 1.0) < 1e-9, f"expected 1.0, got {result['mAP']}"

    dets = [{"image_id": 1, "category": "car", "score": 0.9, "box": [100, 100, 110, 110]}]
    gts = [{"image_id": 1, "category": "car", "box": [0, 0, 10, 10]}]
    result = compute_map(dets, gts)
    assert abs(result["mAP"] - 0.0) < 1e-9, f"expected 0.0, got {result['mAP']}"

    dets = [
        {"image_id": 1, "category": "car", "score": 0.9, "box": [0, 0, 10, 10]},
        {"image_id": 1, "category": "car", "score": 0.1, "box": [200, 200, 210, 210]},
    ]
    gts = [
        {"image_id": 1, "category": "car", "box": [0, 0, 10, 10]},
        {"image_id": 1, "category": "car", "box": [50, 50, 60, 60]},
    ]
    result = compute_map(dets, gts)
    assert abs(result["mAP"] - 0.5) < 1e-9, f"expected 0.5, got {result['mAP']}"

    dets = [{"image_id": 1, "category": "car", "score": 0.9, "box": [0, 0, 10, 10]}]
    gts = [{"image_id": 1, "category": "car", "box": [6, 0, 16, 10]}]  # IoU = 4/16 = 0.25
    result = compute_map(dets, gts)
    assert abs(result["mAP"] - 0.0) < 1e-9, f"expected 0.0 (IoU below threshold), got {result['mAP']}"

    dets = [
        {"image_id": 1, "category": "car", "score": 0.9, "box": [0, 0, 10, 10]},
    ]
    gts = [
        {"image_id": 1, "category": "car", "box": [0, 0, 10, 10]},
        {"image_id": 1, "category": "pedestrian", "box": [50, 50, 60, 60]},
    ]
    result = compute_map(dets, gts)
    assert abs(result["mAP"] - 0.5) < 1e-9, f"expected 0.5, got {result['mAP']}"
    assert abs(result["per_category"]["car"] - 1.0) < 1e-9
    assert abs(result["per_category"]["pedestrian"] - 0.0) < 1e-9

    print("experiments/map_utils.py self-test: all 5 checks PASSED")


if __name__ == "__main__":
    _self_test()