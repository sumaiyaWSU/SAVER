from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import winners_model_DB as WDB

_spec = importlib.util.spec_from_file_location(
    "chain_of_thought", os.path.join(REPO_ROOT, "Chain-of-thought.py")
)
_cot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cot)

TOL = 1e-4

STEP3_MARKERS = [
    ("risk", "--- Step 3 (1) Risk Assessment ---"),
    ("confidence_policy", "--- Step 3 (2)(1) Confidence Policy ---"),
    ("latency_budget_policy", "--- Step 3 (2)(2) Latency Budget Policy ---"),
    ("shortlist_final_decision", "--- Step 3 (3)+(4) Shortlist Candidates & Final Decision ---"),
]

_CONFIDENCE_RAINY_SNOW = (0.65, 0.75)
_CONFIDENCE_DEFAULT = (0.70, 0.75)
_LATENCY_TABLE = {"Downtown": (200.0, 500.0), "Highway": (50.0, 150.0)}


def _isclose(a: Any, b: Any, tol: float = TOL) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def split_step3_raw(raw_text: str) -> Optional[Dict[str, str]]:
    """Splits Step 3's concatenated raw text (built by
    Chain-of-thought.py's run_scene_aware_decision) back into each
    sub-call's own raw completion. Returns None for an Adaptive-Duration-
    reused frame (Step 3 skipped entirely that frame)
    or any trace whose raw text doesn't contain all four markers."""
    positions = []
    for key, marker in STEP3_MARKERS:
        idx = raw_text.find(marker)
        if idx == -1:
            return None
        positions.append((key, idx, idx + len(marker)))
    positions.sort(key=lambda t: t[1])
    out = {}
    for i, (key, _start, content_start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(raw_text)
        out[key] = raw_text[content_start:end].strip()
    return out


def _confidence_base(weather_key: str):
    if weather_key in ("Rainy", "Snow"):
        return _CONFIDENCE_RAINY_SNOW
    return _CONFIDENCE_DEFAULT


def _verify_step2(analysis_json: Dict[str, Any]) -> bool:
    try:
        diff = analysis_json["difficulty"]
        weights, picked, terms, D = diff["weights"], diff["picked"], diff["terms"], diff["D"]
        expected_terms = {
            "ww_times_Ew": weights["ww"] * picked["Ew"],
            "wt_times_Et": weights["wt"] * picked["Et"],
            "wr_times_Er": weights["wr"] * picked["Er"],
            "wl_times_El": weights["wl"] * picked["El"],
        }
        if any(not _isclose(terms[k], v) for k, v in expected_terms.items()):
            return False
        if not _isclose(D, sum(expected_terms.values())):
            return False

        comp = analysis_json["complexity"]
        per_object, N, sum_p = comp["per_object"], comp["N"], comp["sum_p"]
        alpha, beta, c_terms, C = comp["alpha"], comp["beta"], comp["terms"], comp["C"]
        if N != len(per_object):
            return False
        if not _isclose(sum_p, sum(float(o["p"]) for o in per_object)):
            return False
        if not _isclose(c_terms["alpha_times_N"], alpha * N):
            return False
        if not _isclose(c_terms["beta_times_sum_p"], beta * sum_p):
            return False
        if not _isclose(C, c_terms["alpha_times_N"] + c_terms["beta_times_sum_p"]):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _verify_risk(risk_json: Dict[str, Any], D_const: float, C_const: float) -> bool:
    try:
        terms, R = risk_json["terms"], risk_json["R"]
        C_norm_expected = C_const / 20.0
        if not _isclose(terms["C_norm"], C_norm_expected):
            return False
        if not _isclose(terms["risk_D_term"], 0.60 * D_const):
            return False
        if not _isclose(terms["risk_C_term"], 0.40 * C_norm_expected):
            return False
        if not _isclose(R, terms["risk_D_term"] + terms["risk_C_term"]):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _verify_confidence_policy(cp_json: Dict[str, Any], R: float, wk: str) -> bool:
    try:
        picked, terms = cp_json["picked"], cp_json["terms"]
        C_min, C_max = cp_json["C_min"], cp_json["C_max"]
        base_low, base_high = _confidence_base(wk)
        if not _isclose(picked["base_low"], base_low):
            return False
        if not _isclose(picked["base_high"], base_high):
            return False
        if not _isclose(terms["conf_min_term"], 0.04 * R):
            return False
        if not _isclose(terms["conf_max_term"], 0.03 * R):
            return False
        if not _isclose(C_min, picked["base_low"] + terms["conf_min_term"]):
            return False
        if not _isclose(C_max, picked["base_high"] + terms["conf_max_term"]):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _verify_latency_budget(lb_json: Dict[str, Any], R: float, sk: str) -> bool:
    try:
        picked, terms, L_budget_ms = lb_json["picked"], lb_json["terms"], lb_json["L_budget_ms"]
        if sk not in _LATENCY_TABLE:
            return False
        L_min, L_max = _LATENCY_TABLE[sk]
        if not _isclose(picked["L_min"], L_min):
            return False
        if not _isclose(picked["L_max"], L_max):
            return False
        if not _isclose(terms["latency_range_term"], (L_max - L_min) * R):
            return False
        if not _isclose(L_budget_ms, picked["L_min"] + terms["latency_range_term"]):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _rebuild_ranked_all(wk: str, sk: str) -> List[Dict[str, Any]]:
    rows = []
    for model_name, per_model in WDB.DATA.items():
        entry = per_model.get(wk, {}).get(sk)
        if entry is None:
            continue
        rows.append({
            "model": model_name,
            "latency": float(entry["latency"]),
            "mAP": float(entry["mAP"]),
            "confidence": WDB.get_confidence(model_name, wk, sk),
        })
    return sorted(rows, key=lambda r: (-r["mAP"], r["latency"]))


def _expected_shortlist_and_final(ranked_all: List[Dict[str, Any]], L_budget_ms: float, C_min: float):
    """Mirrors Chain-of-thought.py's _expected_shortlist_and_final. Per the
    paper (Sec. III.B.3 (4) Final Decision), candidate set K is by
    construction already latency+confidence feasible; when K is empty,
    SAVER falls back to the single FASTEST model overall, not the
    highest-mAP one."""
    feasible = [r for r in ranked_all
                if r["latency"] <= L_budget_ms and (r["confidence"] is None or r["confidence"] >= C_min)]
    if feasible:
        max_map = max(r["mAP"] for r in feasible)
        tied = [r for r in feasible if _isclose(r["mAP"], max_map, 1e-9)]
        return "IN_BAND", feasible[:3], min(tied, key=lambda r: r["latency"])
    if ranked_all:
        fastest = min(ranked_all, key=lambda r: r["latency"])
        return "FALLBACK", [fastest], fastest
    return "FALLBACK", [], None


def _verify_shortlist_final(shortlist_json: Dict[str, Any], ranked_all: List[Dict[str, Any]],
                             L_budget_ms: float, C_min: float) -> bool:
    try:
        used_mode = shortlist_json["used_mode"]
        shortlist_detail = shortlist_json["shortlist_detail"]
        policy = shortlist_json["policy"]
        expected_mode, expected_shortlist, expected_final = _expected_shortlist_and_final(
            ranked_all, L_budget_ms, C_min
        )
        if used_mode != expected_mode:
            return False
        if policy["K"] != len(shortlist_detail):
            return False
        if [r["model"] for r in shortlist_detail] != [r["model"] for r in expected_shortlist]:
            return False
        if expected_final is None or policy["Final model"] != expected_final["model"]:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def extract_step1_example(trace: Dict[str, Any], images_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    step1 = trace.get("step1_scene_context_extraction")
    if not step1 or not step1.get("raw"):
        return None
    desc_json = step1.get("json")
    if not isinstance(desc_json, dict):
        return None
    env = desc_json.get("environment")
    if not isinstance(env, dict) or not all(k in env for k in ("weather", "time", "road", "lane")):
        return None
    if not isinstance(desc_json.get("critical_objects"), list):
        return None

    image_name = (trace.get("meta") or {}).get("image")
    image_path = os.path.join(images_dir, image_name) if images_dir and image_name else None
    if image_path and not os.path.exists(image_path):
        image_path = None

    return {
        "step": "step1_scene_context_extraction",
        "prompt": _cot.build_scene_context_extraction_prompt(),
        "image": image_path or image_name,
        "image_resolved": image_path is not None,
        "completion": step1["raw"].strip(),
        "prompt_exact": True,
    }


def extract_step2_example(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    step1 = trace.get("step1_scene_context_extraction")
    step2 = trace.get("step2_scene_context_scoring")
    if not step1 or not step2 or not step1.get("raw") or not step2.get("raw"):
        return None
    analysis_json = step2.get("json")
    if not isinstance(analysis_json, dict) or not _verify_step2(analysis_json):
        return None

    return {
        "step": "step2_scene_context_scoring",
        "prompt": _cot.build_scene_context_scoring_prompt(step1["raw"]),
        "completion": step2["raw"].strip(),
        "prompt_exact": True,
    }


def extract_step3_examples(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    step1 = trace.get("step1_scene_context_extraction")
    step2 = trace.get("step2_scene_context_scoring")
    step3 = trace.get("step3_scene_aware_decision")
    if not step1 or not step2 or not step3 or not step2.get("json"):
        return []

    raw_parts = split_step3_raw(step3.get("raw", ""))
    if raw_parts is None:
        return []

    audit = (step3.get("json") or {}).get("audit")
    if not isinstance(audit, dict):
        return []

    try:
        D_const = float(step2["json"]["difficulty"]["D"])
        C_const = float(step2["json"]["complexity"]["C"])
        wk, sk = audit["weather_key"], audit["scene_key"]
        R = float(audit["R"])
        C_min = float(audit["C_min"])
        C_max = float(audit["C_max"])
        L_budget_ms = float(audit["L_budget_ms"])
    except (KeyError, TypeError, ValueError):
        return []

    examples: List[Dict[str, Any]] = []

    risk_json = _cot.try_parse_json(raw_parts["risk"])
    if risk_json and _verify_risk(risk_json, D_const, C_const):
        examples.append({
            "step": "step3_risk_assessment",
            "prompt": _cot.build_risk_assessment_prompt(D_const, C_const),
            "completion": raw_parts["risk"],
            "prompt_exact": True,
        })

    cp_json = _cot.try_parse_json(raw_parts["confidence_policy"])
    if cp_json and _verify_confidence_policy(cp_json, R, wk):
        examples.append({
            "step": "step3_confidence_policy",
            "prompt": _cot.build_confidence_policy_prompt(R, wk),
            "completion": raw_parts["confidence_policy"],
            "prompt_exact": True,
        })

    lb_json = _cot.try_parse_json(raw_parts["latency_budget_policy"])
    if lb_json and _verify_latency_budget(lb_json, R, sk):
        examples.append({
            "step": "step3_latency_budget_policy",
            "prompt": _cot.build_latency_budget_prompt(R, sk),
            "completion": raw_parts["latency_budget_policy"],
            "prompt_exact": True,
        })

    shortlist_json = _cot.try_parse_json(raw_parts["shortlist_final_decision"])
    if shortlist_json:
        ranked_all = _rebuild_ranked_all(wk, sk)
        if ranked_all and _verify_shortlist_final(shortlist_json, ranked_all, L_budget_ms, C_min):
            ranked_all_json = json.dumps(ranked_all, ensure_ascii=False)
            prompt = _cot.build_shortlist_and_final_decision_prompt(
                step1["raw"], step2["raw"], D_const, C_const, wk, sk,
                R, C_min, C_max, L_budget_ms, ranked_all_json,
            )
            examples.append({
                "step": "step3_shortlist_and_final_decision",
                "prompt": prompt,
                "completion": raw_parts["shortlist_final_decision"],
                "prompt_exact": False,
            })

    return examples


STEP_NAMES = [
    "step1_scene_context_extraction",
    "step2_scene_context_scoring",
    "step3_risk_assessment",
    "step3_confidence_policy",
    "step3_latency_budget_policy",
    "step3_shortlist_and_final_decision",
]


def export(results_dir: str, output_path: str, images_dir: Optional[str] = None) -> Dict[str, Any]:
    counts = {name: 0 for name in STEP_NAMES}
    total_files = 0
    skipped_files = 0
    examples: List[Dict[str, Any]] = []

    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        total_files += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped_files += 1
            continue
        if not isinstance(trace, dict) or "step1_scene_context_extraction" not in trace:
            skipped_files += 1
            continue

        step1_example = extract_step1_example(trace, images_dir)
        if step1_example:
            step1_example["source_file"] = path
            examples.append(step1_example)
            counts["step1_scene_context_extraction"] += 1

        step2_example = extract_step2_example(trace)
        if step2_example:
            step2_example["source_file"] = path
            examples.append(step2_example)
            counts["step2_scene_context_scoring"] += 1

        for ex in extract_step3_examples(trace):
            ex["source_file"] = path
            examples.append(ex)
            counts[ex["step"]] += 1

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return {
        "total_files": total_files,
        "skipped_files": skipped_files,
        "examples_written": len(examples),
        "counts": counts,
    }


def _self_test() -> None:
    """Synthetic fixtures (no real results/*.json needed) verifying the
    self-check-gated extraction: a correct example is exported, and one
    with each specific kind of drifted arithmetic (the same failure modes
    documented in Chain-of-thought.py's run_risk_assessment/
    run_confidence_policy docstrings) is rejected."""
    good_risk = {"terms": {"C_norm": 0.1, "risk_D_term": 0.3, "risk_C_term": 0.04}, "R": 0.34}
    assert _verify_risk(good_risk, D_const=0.5, C_const=2.0)

    drifted_risk = dict(good_risk)
    drifted_risk["R"] = 0.6  # doesn't equal terms sum; the exact bug this pipeline was built to catch
    assert not _verify_risk(drifted_risk, D_const=0.5, C_const=2.0)

    good_cp = {"picked": {"base_low": 0.70, "base_high": 0.75},
               "terms": {"conf_min_term": 0.012, "conf_max_term": 0.009},
               "C_min": 0.712, "C_max": 0.759}
    assert _verify_confidence_policy(good_cp, R=0.3, wk="Sunny")

    leaked_cp = dict(good_cp)
    leaked_cp["terms"] = {"conf_min_term": 0.0294, "conf_max_term": 0.009}  # R=0.735, not 0.3
    assert not _verify_confidence_policy(leaked_cp, R=0.3, wk="Sunny")

    good_lb = {"picked": {"L_min": 50.0, "L_max": 150.0}, "terms": {"latency_range_term": 30.0},
               "L_budget_ms": 80.0}
    assert _verify_latency_budget(good_lb, R=0.3, sk="Highway")

    wrong_lb = dict(good_lb)
    wrong_lb["L_budget_ms"] = 100.0  # (L_max - L_min) alone, forgot + L_min 
    assert not _verify_latency_budget(wrong_lb, R=0.3, sk="Highway")

    good_step2 = {
        "difficulty": {
            "weights": {"ww": 0.3, "wt": 0.2, "wr": 0.3, "wl": 0.2},
            "picked": {"Ew": 0.7, "Et": 0.2, "Er": 0.3, "El": 0.5},
            "terms": {"ww_times_Ew": 0.21, "wt_times_Et": 0.04, "wr_times_Er": 0.09, "wl_times_El": 0.1},
            "D": 0.44,
        },
        "complexity": {
            "per_object": [{"class": "pedestrian", "p": 0.9}],
            "N": 1, "sum_p": 0.9, "alpha": 1.0, "beta": 2.0,
            "terms": {"alpha_times_N": 1.0, "beta_times_sum_p": 1.8}, "C": 2.8,
        },
    }
    assert _verify_step2(good_step2)

    bad_step2 = json.loads(json.dumps(good_step2))
    bad_step2["complexity"]["N"] = 2  # doesn't match len(per_object)
    assert not _verify_step2(bad_step2)

    ranked = [
        {"model": "A", "latency": 50.0, "mAP": 0.8, "confidence": 0.9},
        {"model": "B", "latency": 60.0, "mAP": 0.8, "confidence": 0.9},
        {"model": "C", "latency": 40.0, "mAP": 0.6, "confidence": None},
    ]
    mode, shortlist, final = _expected_shortlist_and_final(ranked, L_budget_ms=100.0, C_min=0.5)
    assert mode == "IN_BAND"
    assert final["model"] == "A"  # tied max mAP with B, A wins on lower latency

    good_shortlist = {"used_mode": "IN_BAND",
                       "shortlist_detail": [{"model": "A"}, {"model": "B"}, {"model": "C"}],
                       "policy": {"K": 3, "Final model": "A"}}
    assert _verify_shortlist_final(good_shortlist, ranked, L_budget_ms=100.0, C_min=0.5)

    wrong_shortlist = json.loads(json.dumps(good_shortlist))
    wrong_shortlist["policy"]["Final model"] = "B"  # faster tie-break picked wrong, or a non-tied swap
    assert not _verify_shortlist_final(wrong_shortlist, ranked, L_budget_ms=100.0, C_min=0.5)

    print("fine-tuning.py self-test: all checks PASSED")


def parse_args():
    p = argparse.ArgumentParser(
        description="Mine results/*.json decision traces into a labeled SFT "
                     "dataset for Qwen2.5-VL fine-tuning."
    )
    p.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"),
                   help="Directory of decision-trace JSON files (searched recursively)")
    p.add_argument("--output", default=os.path.join(REPO_ROOT, "sft_dataset.jsonl"),
                   help="Output JSONL path, one example per line")
    p.add_argument("--images-dir", default=None,
                   help="Directory containing the original input images, for resolving Step 1's "
                        "image path (meta.image in each trace); Step 1 examples are still exported "
                        "without it, with image_resolved=false")
    p.add_argument("--self-test", action="store_true",
                   help="Run the synthetic self-test (no results/*.json needed) and exit")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        _self_test()
        return

    summary = export(args.results_dir, args.output, images_dir=args.images_dir)
    print(f"[fine-tuning] scanned {summary['total_files']} file(s) under {args.results_dir} "
          f"({summary['skipped_files']} not a decision trace, skipped)")
    print(f"[fine-tuning] wrote {summary['examples_written']} example(s) -> {args.output}")
    for name in STEP_NAMES:
        print(f"  {name}: {summary['counts'][name]}")


if __name__ == "__main__":
    main()