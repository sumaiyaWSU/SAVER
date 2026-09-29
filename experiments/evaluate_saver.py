from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import yaml

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXPERIMENTS_DIR)
for _p in (REPO_ROOT, EXPERIMENTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import detector_runner as DR
import duration_tracker as DT
import winners_model_DB as WDB
from dataset_loader import Frame, load_dataset, summarize
from map_utils import compute_map

_spec = importlib.util.spec_from_file_location(
    "rl_integrated_cot", os.path.join(REPO_ROOT, "RL_integrated-CoT.py")
)
_rlcot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rlcot)


LATENCY_TIERS = {
    "Highway": [65.0, 85.0, 115.0],
    "Downtown": [145.0, 225.0, 325.0],
}

DEFAULT_SCORE_THRESHOLD = DR.DEFAULT_SCORE_THRESHOLD


def _best_static_detector(weather: str, scene: str, tier_ms: float) -> Optional[str]:
    """Table VIII's 'Best Static Detector' column: the highest-mAP single
    model from the Winner Model Database whose PROFILED latency (Table VI)
    fits this tier; ties broken by lower profiled latency. Returns None if
    no model's profile is available for this weather/scene."""
    candidates = []
    for model_name, per_model in WDB.DATA.items():
        entry = per_model.get(weather, {}).get(scene)
        if entry is None:
            continue
        candidates.append((model_name, float(entry["mAP"]), float(entry["latency"])))
    if not candidates:
        return None
    feasible = [c for c in candidates if c[2] <= tier_ms]
    pool = feasible if feasible else candidates
    pool.sort(key=lambda c: (-c[1], c[2]))
    return pool[0][0]


def _run_static_detector(model_name: Optional[str], image_path: str,
                          score_threshold: float) -> Optional[DR.DetectionRunResult]:
    """Runs the single named 'Best Static Detector' for real, degrading to
    None (frame skipped for this cell) rather than crashing the whole
    evaluation if that particular model has no usable backend in this
    environment (e.g. Sparse-RCNN without detectron2 installed)."""
    if model_name is None:
        return None
    try:
        return DR.run_detector(model_name, image_path, score_threshold=score_threshold)
    except DR.DetectorUnavailableError as exc:
        print(f"[evaluate_saver] Best Static Detector '{model_name}' unavailable, "
              f"skipping this frame for this cell: {exc}")
        return None


def _select_within_tier(shortlist_detail: List[Dict[str, Any]], tier_ms: float) -> List[Dict[str, Any]]:
    """Downstream, read-only re-selection: the CoT's own shortlist_detail
    is already ordered mAP-desc/latency-asc (see run_shortlist_and_final_decision).
    Returns the tier-feasible rows in that same order, or -- if none fit --
    the single fastest row of the shortlist (same FALLBACK convention Step
    3 itself uses internally when nothing meets its own L_budget_ms: per
    the paper's Sec. III.B.3 (4) Final Decision, when nothing is feasible
    SAVER falls back to the fastest model, not the highest-mAP one)."""
    feasible = [row for row in shortlist_detail if float(row.get("latency", 0.0)) <= tier_ms]
    if feasible:
        return feasible
    if not shortlist_detail:
        return []
    return [min(shortlist_detail, key=lambda row: float(row.get("latency", 0.0)))]


class _DurationState:
    """Per-sequence Adaptive Duration bookkeeping (Sec. III.B.2), kept
    in-memory and keyed by sequence_id -- unlike RL_integrated-CoT.py's
    main(), which persists one global duration_state.json for a single
    continuous stream of separate CLI invocations, an offline dataset
    evaluation processes many independent sequences (possibly interleaved
    across weather/scene buckets), so state must not leak between them."""

    def __init__(self):
        self._by_sequence: Dict[str, Dict[str, Any]] = {}

    def get(self, sequence_id: str) -> Optional[Dict[str, Any]]:
        return self._by_sequence.get(sequence_id)

    def set(self, sequence_id: str, state: Dict[str, Any]) -> None:
        self._by_sequence[sequence_id] = state


def _run_cot_for_frame(model, frame: Frame, duration_state: _DurationState,
                        duration_min_frames: int, duration_max_frames: int) -> Dict[str, Any]:
    """Runs Steps 1-3 (Scene Context Extraction, Scene Context Scoring,
    Scene-aware Decision) plus the Adaptive Duration reuse decision for one
    frame, exactly mirroring RL_integrated-CoT.py's main() -- only the
    duration-state storage (per-sequence, in-memory) differs, since main()
    assumes one continuous CLI-invocation-per-frame stream. Returns a dict
    with weather_key, scene_key, D, C, R, C_min, C_max, L_budget_ms,
    shortlist_detail (list of {"model","latency","mAP","confidence"}), and
    llm_final_model (Step 3's own unconditional choice, before any RL
    layer or tier re-selection is applied)."""

    class _Args:
        image = frame.image_path

    args = _Args()

    desc_json, desc_text, _, _ = _rlcot.run_scene_context_extraction(model, args)
    analysis_json, analysis_text, _, _ = _rlcot.run_scene_context_scoring(model, args, desc_text)

    desc_env = (desc_json or {}).get("environment", {})
    _weather_str = str(desc_env.get("weather", "")).lower().strip()
    _road_str = str(desc_env.get("road", "")).lower().strip()

    wk = WDB.WEATHER_MAP.get(_weather_str)
    if wk is None:
        if "sand" in _weather_str or "dust" in _weather_str:
            wk = WDB.WEATHER_MAP.get("sand") or "Sand"
        elif "fog" in _weather_str or "mist" in _weather_str:
            wk = WDB.WEATHER_MAP.get("fog") or "Foggy"
        elif "overcast" in _weather_str or "cloud" in _weather_str:
            wk = "Sunny"
        elif "snow" in _weather_str or "blizzard" in _weather_str:
            wk = WDB.WEATHER_MAP.get("snow") or "Snow"
        elif "rain" in _weather_str or "wet" in _weather_str:
            wk = WDB.WEATHER_MAP.get("rain") or "Rainy"
    if wk is None:
        wk = "Sunny"
    sk = "Highway" if ("highway" in _road_str or "ramp" in _road_str) else "Downtown"

    slice_rows_py = []
    for model_name, per_model in WDB.DATA.items():
        try:
            entry = per_model[wk][sk]
            slice_rows_py.append({
                "model": model_name,
                "latency": float(entry["latency"]),
                "mAP": float(entry["mAP"]),
                "confidence": WDB.get_confidence(model_name, wk, sk),
            })
        except KeyError:
            continue
    ranked_all_py = sorted(slice_rows_py, key=lambda r: (-r["mAP"], r["latency"]))
    ranked_all_json = json.dumps(ranked_all_py, ensure_ascii=False)

    D_const = float((analysis_json.get("difficulty") or {}).get("D") or 0.0)
    C_const = float((analysis_json.get("complexity") or {}).get("C") or 0.0)

    if frame.is_new_sequence:
        prior = None
    else:
        prior = duration_state.get(frame.sequence_id)

    _time_of_day = str(desc_env.get("time", "")).strip()
    _prev_C_max = float(prior.get("C_max_seen", 0.0)) if prior else 0.0
    C_max_seen = max(_prev_C_max, C_const)

    feature_vector_now = DT.build_feature_vector(wk, sk, _time_of_day, D_const, C_const, C_max_seen)
    prev_feature_vector = prior.get("feature_vector") if prior else None
    similarity_now = DT.cosine_similarity(feature_vector_now, prev_feature_vector or [])

    _remaining_duration = int(prior.get("remaining_duration", 0)) if prior else 0
    _stored_rl_state = prior.get("rl_state") if prior else None
    reuse_active = (
        prior is not None
        and _remaining_duration > 0
        and _stored_rl_state is not None
        and prior.get("active_shortlist") is not None
    )

    if reuse_active:
        shortlist_detail = prior["active_shortlist"]
        llm_final_model = prior["active_llm_final_model"]
        R = _stored_rl_state["R"]
        C_min = _stored_rl_state["C_min"]
        C_max = _stored_rl_state["C_max"]
        L_budget_ms = _stored_rl_state["L_budget"]
    else:
        decision_json, _, _, _ = _rlcot.run_scene_aware_decision(
            model, args, desc_text, analysis_text, D_const, C_const, wk, sk, ranked_all_json
        )
        _audit = decision_json["audit"]
        shortlist_detail = _audit.get("shortlist_detail", [])
        llm_final_model = decision_json["policy"]["Final model"]
        R = float(_audit["R"])
        C_min = float(_audit["C_min"])
        C_max = float(_audit["C_max"])
        L_budget_ms = float(_audit["L_budget_ms"])

    rl_state = {
        "weather": wk, "scene": sk, "D": D_const, "C": C_const,
        "R": R, "C_min": C_min, "C_max": C_max, "L_budget": L_budget_ms,
        "shortlist": shortlist_detail,
    }

    if reuse_active:
        _next_remaining = _remaining_duration - 1
    else:
        _c_hat_now = (C_const / C_max_seen) if C_max_seen > 0 else 0.0
        _new_tau = DT.compute_duration(
            similarity_now, _c_hat_now,
            tau_min=duration_min_frames, tau_max=duration_max_frames,
        )
        _next_remaining = _new_tau - 1

    duration_state.set(frame.sequence_id, {
        "feature_vector": feature_vector_now,
        "remaining_duration": _next_remaining,
        "C_max_seen": C_max_seen,
        "rl_state": rl_state,
        "active_shortlist": shortlist_detail,
        "active_llm_final_model": llm_final_model,
    })

    return {
        "weather_key": wk, "scene_key": sk,
        "D": D_const, "C": C_const, "R": R, "C_min": C_min, "C_max": C_max,
        "L_budget_ms": L_budget_ms, "shortlist_detail": shortlist_detail,
        "llm_final_model": llm_final_model, "reused": reuse_active,
    }


def _gt_to_eval_rows(frame: Frame) -> List[Dict[str, Any]]:
    return [{"image_id": frame.image_path, "category": g["category"], "box": g["box"]}
            for g in frame.ground_truth]


def _detections_to_eval_rows(image_id: str, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"image_id": image_id, "category": str(d["label"]).strip().lower(),
              "score": d["score"], "box": d["box"]} for d in detections]


class _BucketAccumulator:
    """Collects real detections + real latencies for one (weather, scene,
    tier, method) cell of Table VIII, across every frame it's applied to."""

    def __init__(self):
        self.detections: List[Dict[str, Any]] = []
        self.ground_truth: List[Dict[str, Any]] = []
        self.latencies_ms: List[float] = []
        self.frames_run = 0
        self.frames_skipped = 0

    def add(self, frame: Frame, exec_result: Optional[DR.DetectionRunResult]) -> None:
        self.ground_truth.extend(_gt_to_eval_rows(frame))
        if exec_result is None:
            self.frames_skipped += 1
            return
        self.frames_run += 1
        self.latencies_ms.append(exec_result.latency_ms)
        self.detections.extend(_detections_to_eval_rows(frame.image_path, exec_result.detections))

    def result(self) -> Dict[str, Any]:
        map_result = compute_map(self.detections, self.ground_truth)
        mean_latency = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None
        return {
            "mAP": map_result["mAP"],
            "latency_ms": mean_latency,
            "frames_run": self.frames_run,
            "frames_skipped": self.frames_skipped,
            "num_ground_truth_boxes": len(self.ground_truth),
        }


def evaluate(dataset_root: str, mode: str, model, score_threshold: float = DEFAULT_SCORE_THRESHOLD,
             duration_min_frames: int = DT.DEFAULT_TAU_MIN, duration_max_frames: int = DT.DEFAULT_TAU_MAX,
             progress: bool = True) -> Dict[str, Any]:
    """Evaluates one mode ("vlm" or "vlm_rl") over every weather/scene
    bucket present under dataset_root, per Table VIII's latency-budget
    tiers. Returns the full results structure (see main()'s --output)."""
    if mode not in ("vlm", "vlm_rl"):
        raise ValueError(f"mode must be 'vlm' or 'vlm_rl', got {mode!r}")

    buckets = load_dataset(dataset_root)
    if not buckets:
        raise RuntimeError(f"No weather/scene buckets found under {dataset_root}. "
                            f"See dataset_loader.py's module docstring for the expected layout.")
    if progress:
        print(f"[evaluate_saver] loaded dataset:\n{summarize(buckets)}")

    # One isolated Q-memory dict per (scene, tier), NOT a single shared dict.
    # make_signature()'s L_budget bucket cuts ([100, 250, 500]) were chosen
    # for the CoT's own internal L_budget_ms range, not for Table VIII's six
    # external tier values -- 65ms/85ms both bucket to "<= 100" and
    # 145ms/225ms both bucket to "<= 250", so a single shared RL_Q would let
    # an execution under one tier silently influence routing for a
    # different tier of the same scene. Keying by (scene, tier) sidesteps
    # that bucket collision entirely, independent of how coarse the shared
    # signature buckets are.
    rl_q_by_tier: Dict[Any, Dict[str, float]] = {}

    results: Dict[str, Any] = {}
    for (weather, scene), frames in sorted(buckets.items()):
        tiers = LATENCY_TIERS[scene]
        static_acc = {tier: _BucketAccumulator() for tier in tiers}
        saver_acc = {tier: _BucketAccumulator() for tier in tiers}
        duration_state = _DurationState()

        for i, frame in enumerate(frames):
            if progress:
                print(f"[evaluate_saver] {weather}/{scene} frame {i + 1}/{len(frames)} "
                      f"(seq={frame.sequence_id}, idx={frame.frame_index}) ...")

            cot = _run_cot_for_frame(model, frame, duration_state, duration_min_frames, duration_max_frames)
            shortlist_detail = cot["shortlist_detail"]

            for tier in tiers:
                static_model = _best_static_detector(weather, scene, tier)
                static_result = _run_static_detector(static_model, frame.image_path, score_threshold)
                static_acc[tier].add(frame, static_result)

                eligible = _select_within_tier(shortlist_detail, tier)
                if not eligible:
                    saver_acc[tier].add(frame, None)
                    continue

                if mode == "vlm":
                    candidate_order = [row["model"] for row in eligible]
                else:
                    tier_key = (scene, tier)
                    if tier_key not in rl_q_by_tier:
                        rl_q_by_tier[tier_key] = {}
                    tier_rl_state = {
                        "weather": cot["weather_key"], "scene": cot["scene_key"],
                        "D": cot["D"], "C": cot["C"], "R": cot["R"],
                        "C_min": cot["C_min"], "C_max": cot["C_max"],
                        "L_budget": tier, "shortlist": eligible,
                    }
                    llm_choice = eligible[0]["model"]
                    _rlcot.RL_Q = rl_q_by_tier[tier_key]
                    rl_decision = _rlcot.rl_policy_select(tier_rl_state, llm_choice)
                    final_model = rl_decision["final_model"]
                    candidate_order = [final_model] + [
                        row["model"] for row in eligible if row["model"] != final_model
                    ]

                exec_result = DR.run_with_fallback(candidate_order, frame.image_path, score_threshold)
                saver_acc[tier].add(frame, exec_result)

                if mode == "vlm_rl" and exec_result is not None:
                    reward, penalty = _rlcot.compute_reward(
                        exec_result.confidence, exec_result.latency_ms, tier
                    )
                    _rlcot.RL_Q = rl_q_by_tier[tier_key]
                    _rlcot.rl_update_after_execution(tier_rl_state, exec_result.model, reward, penalty)

        results[f"{weather}/{scene}"] = {
            "weather": weather, "scene": scene,
            "tiers": {
                f"<{int(tier)}ms": {
                    "best_static_detector": static_acc[tier].result(),
                    f"saver_{mode}": saver_acc[tier].result(),
                }
                for tier in tiers
            },
        }

    return {
        "meta": {
            "dataset": os.path.abspath(dataset_root),
            "mode": mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latency_tiers": LATENCY_TIERS,
            "note": "mAP/latency in every cell are measured from real routed "
                    "detector executions against ground truth, not copied "
                    "from winners_model_DB.py's offline profiles.",
        },
        "results": results,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="SAVER Quantitative Evaluation -- Table VIII. Generates "
                     "mAP/latency per weather x scene x latency-budget tier "
                     "from real routed detector executions."
    )
    p.add_argument("--dataset", required=True, help="Dataset root (see dataset_loader.py)")
    p.add_argument("--mode", required=True, choices=["vlm", "vlm_rl"],
                   help="'vlm': Steps 1-3 only, Step 3's own Final model used "
                        "unconditionally (no RL). 'vlm_rl': full pipeline, "
                        "Adaptive RL Module routes among the tier-feasible "
                        "shortlist and Q-memory updates after each execution.")
    p.add_argument("--output", required=True, help="Path to write the results JSON to")
    p.add_argument("--model", default="qwen2.5-7b", choices=["qwen2.5-7b", "qwen2.5-72b"])
    p.add_argument("--config", default=os.path.join(REPO_ROOT, "config.yaml"))
    p.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    p.add_argument("--duration-min-frames", type=int, default=DT.DEFAULT_TAU_MIN)
    p.add_argument("--duration-max-frames", type=int, default=DT.DEFAULT_TAU_MAX)
    return p.parse_args()


def main():
    args = parse_args()

    from vlm import ModelHandler
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model = ModelHandler(args.model, cfg)
    model.initialize_model()

    out = evaluate(
        args.dataset, args.mode, model,
        score_threshold=args.score_threshold,
        duration_min_frames=args.duration_min_frames,
        duration_max_frames=args.duration_max_frames,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[evaluate_saver] saved -> {args.output}")


if __name__ == "__main__":
    main()