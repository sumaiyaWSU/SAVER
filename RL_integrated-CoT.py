import argparse
import importlib.util
import json
import os
import sys
import time
import yaml
from typing import Any, Dict

import winners_model_DB as WDB
import detector_runner as DR
import duration_tracker as DT


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

_cot_spec = importlib.util.spec_from_file_location(
    "chain_of_thought", os.path.join(REPO_ROOT, "Chain-of-thought.py")
)
_cot = importlib.util.module_from_spec(_cot_spec)
_cot_spec.loader.exec_module(_cot)

run_scene_context_extraction = _cot.run_scene_context_extraction
run_scene_context_scoring = _cot.run_scene_context_scoring
run_scene_aware_decision = _cot.run_scene_aware_decision
strip_md_fences = _cot.strip_md_fences
try_parse_json = _cot.try_parse_json


RL_Q = {}

RL_MIN_CONFIDENCE = 0.70

RL_LEARNING_RATE = 0.1

RL_LATENCY_PENALTY_BETA = 0.5

def _bucket(val: float, cuts):

    for c in cuts:
        if val <= c:
            return f"<= {c}"
    return f"> {cuts[-1]}"

def make_signature(weather: str,
                   scene: str,
                   D: float,
                   C: float,
                   R: float,
                   C_min: float,
                   C_max: float,
                   L_budget: float):
    Db = _bucket(D, [0.3, 0.6, 1.0])
    Cb = _bucket(C, [1.0, 6.0, 12.0])
    Rb = _bucket(R, [0.3, 0.6, 1.0])
    CMinb = _bucket(C_min, [0.65, 0.72, 0.80])
    CMaxb = _bucket(C_max, [0.65, 0.72, 0.80])
    Bb = _bucket(L_budget, [100, 250, 500])
    return (weather, scene, Db, Cb, Rb, CMinb, CMaxb, Bb)


def _signature_for_state(rl_state: Dict[str, Any]):
    return make_signature(
        rl_state["weather"],
        rl_state["scene"],
        rl_state["D"],
        rl_state["C"],
        rl_state["R"],
        rl_state["C_min"],
        rl_state["C_max"],
        rl_state["L_budget"]
    )


def rl_policy_select(rl_state: Dict[str, Any], llm_choice: str):

    sig = _signature_for_state(rl_state)

    shortlist_rows = rl_state.get("shortlist", [])
    shortlist_models = [row.get("model") for row in shortlist_rows]

    scored_opts = []
    for m in shortlist_models:
        key_str = json.dumps({"sig": sig, "model": m})
        q_val = RL_Q.get(key_str, 0.0)
        scored_opts.append((m, q_val))

    if scored_opts and max(q for _, q in scored_opts) > 0.0:
        best_model, best_q = max(scored_opts, key=lambda x: x[1])
    else:
        best_model, best_q = llm_choice, 0.0

    return {
        "final_model": best_model,
        "chosen_q": best_q,
        "llm_choice": llm_choice,
        "signature": sig
    }


def rl_update_after_execution(rl_state: Dict[str, Any],
                              chosen_model: str,
                              reward: float,
                              penalty: float = 0.0,
                              alpha: float = RL_LEARNING_RATE):
    sig = _signature_for_state(rl_state)

    key_str = json.dumps({"sig": sig, "model": chosen_model})

    old_q = RL_Q.get(key_str, 0.0)
    RL_Q[key_str] = old_q + alpha * ((reward - penalty) - old_q)


def compute_reward(confidence: float, latency_ms: float, L_budget: float,
                    min_confidence: float = RL_MIN_CONFIDENCE,
                    beta: float = RL_LATENCY_PENALTY_BETA):
    """Paper Eq. 1 + the update-side latency penalty (Sec. III.C.3):
        S_conf = 1 if C_t >= C_min^RL else 0
        S_lat  = 1 if L_t <= L_budget else 0
        r_t    = S_conf * S_lat
        c_t    = beta * max(0, (L_t - L_budget) / L_budget)
    """
    s_conf = 1.0 if confidence >= min_confidence else 0.0
    s_lat = 1.0 if latency_ms <= L_budget else 0.0
    reward = s_conf * s_lat
    penalty = beta * max(0.0, (latency_ms - L_budget) / L_budget) if L_budget > 0 else 0.0
    return reward, penalty


def parse_args():
    p = argparse.ArgumentParser(
        description="SAVER: Chain-of-thought.py's Steps 1-3 plus the Adaptive "
                     "RL Module (Q-memory routing + reward update) and the "
                     "Execution Block."
    )
    p.add_argument("--image", required=True, help="Path to the front-view image")
    p.add_argument("--model", default="qwen2.5-7b",
                   choices=["qwen2.5-7b", "qwen2.5-72b"], help="Choose Qwen model")
    p.add_argument("--config", default=os.path.join(REPO_ROOT, "config.yaml"), help="Path to config.yaml (HF token/cache)")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "scene_desc_analysis.json"), help="Output JSON")
    p.add_argument("--frames-dir", default=os.path.join(REPO_ROOT, "output_frames"),
                   help="Directory to save the executed detector's annotated output frame into")
    p.add_argument("--new-sequence", action="store_true",
                   help="Treat this frame as the start of a new, unrelated frame sequence: "
                        "clears any carried-over Adaptive Duration state (Sec. III.B.2) instead "
                        "of comparing this frame against a prior one from a different clip.")
    p.add_argument("--duration-min-frames", type=int, default=DT.DEFAULT_TAU_MIN,
                   help="tau_min for the Adaptive Duration formula (frames)")
    p.add_argument("--duration-max-frames", type=int, default=DT.DEFAULT_TAU_MAX,
                   help="tau_max for the Adaptive Duration formula (frames)")
    return p.parse_args()


def main():
    args = parse_args()
    global RL_Q
    rl_memory_path = os.path.join(REPO_ROOT, "rl_memory.json")
    if os.path.exists(rl_memory_path):
        with open(rl_memory_path, "r") as fmem:
            RL_Q = json.load(fmem)
    else:
        RL_Q = {}


    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model = _cot.ModelHandler(args.model, cfg)
    model.initialize_model()

    print(f"\n Calling Qwen model handler: {args.model}")
    print(f"  Image: {args.image}\n")

    desc_json, desc_text, desc_tokens, desc_time = run_scene_context_extraction(model, args)

    analysis_json, analysis_text, analysis_tokens, analysis_time = run_scene_context_scoring(model, args, desc_text)


    desc_env = (desc_json or {}).get("environment", {})
    _weather_str = str(desc_env.get("weather", "")).lower().strip()
    _road_str    = str(desc_env.get("road", "")).lower().strip()


    wk = WDB.WEATHER_MAP.get(_weather_str)

    if wk is None:
        if "sand" in _weather_str or "dust" in _weather_str:
            wk = WDB.WEATHER_MAP.get("sand") or WDB.WEATHER_MAP.get("sandstorm") or "Sand storm"
        elif "fog" in _weather_str or "mist" in _weather_str:
            wk = WDB.WEATHER_MAP.get("fog") or "Foggy"
        elif "overcast" in _weather_str or "cloud" in _weather_str:
            wk = WDB.WEATHER_MAP.get("cloudy") or "Cloudy"
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

    D_const = float((analysis_json.get("difficulty") or {}).get("D") or 0.0)
    C_const = float((analysis_json.get("complexity") or {}).get("C") or 0.0)

    if args.new_sequence:
        DT.clear_state()
        _duration_state = None
    else:
        _duration_state = DT.load_state()

    _time_of_day = str(desc_env.get("time", "")).strip()
    _prev_C_max = float(_duration_state.get("C_max_seen", 0.0)) if _duration_state else 0.0
    C_max_seen = max(_prev_C_max, C_const)

    feature_vector_now = DT.build_feature_vector(wk, sk, _time_of_day, D_const, C_const, C_max_seen)
    prev_feature_vector = _duration_state.get("feature_vector") if _duration_state else None
    similarity_now = DT.cosine_similarity(feature_vector_now, prev_feature_vector or [])

    _remaining_duration = int(_duration_state.get("remaining_duration", 0)) if _duration_state else 0
    _stored_rl_state = _duration_state.get("rl_state") if _duration_state else None
    reuse_active_model = (
        _duration_state is not None
        and _remaining_duration > 0
        and _stored_rl_state is not None
        and _duration_state.get("active_model")
    )

    print(f"\n[DURATION] similarity(S_hat_t)={similarity_now:.4f}  "
          f"C_hat_t={(C_const / C_max_seen if C_max_seen > 0 else 0.0):.4f}  "
          f"remaining_duration={_remaining_duration}  "
          f"decision={'REUSE ' + str(_duration_state.get('active_model')) if reuse_active_model else 'REEVALUATE'}")

    ranked_all_py = sorted(slice_rows_py, key=lambda r: (-r["mAP"], r["latency"]))
    ranked_all_json = json.dumps(ranked_all_py, ensure_ascii=False)

    if reuse_active_model:
        _reused_model = _duration_state["active_model"]
        _reused_shortlist = _stored_rl_state.get("shortlist", [])
        decision_json = {
            "audit": {
                "D": _stored_rl_state["D"], "C": _stored_rl_state["C"],
                "weather_key": _stored_rl_state["weather"], "scene_key": _stored_rl_state["scene"],
                "R": _stored_rl_state["R"], "C_min": _stored_rl_state["C_min"],
                "C_max": _stored_rl_state["C_max"], "L_budget_ms": _stored_rl_state["L_budget"],
                "used_mode": _stored_rl_state.get("used_mode", "REUSED"),
                "shortlist_detail": _reused_shortlist,
            },
            "policy": {
                "K": len(_reused_shortlist),
                "candidates": [row.get("model") for row in _reused_shortlist],
                "Final model": _reused_model,
            },
            "reasoning": (
                f"Adaptive Duration (Sec. III.B.2): reusing {_reused_model} from the last "
                f"reevaluation ({_remaining_duration} frame(s) of duration remaining "
                f"before this one; S_hat_t={similarity_now:.3f})."
            ),
        }
        decision_text = "[Adaptive Duration: Step 3 (Scene-aware Decision) skipped this frame, model reused from prior decision]"
        decision_tokens = {"input": 0, "output": 0}
        decision_time = 0.0
    else:
        decision_json, decision_text, decision_tokens, decision_time = run_scene_aware_decision(
            model, args, desc_text, analysis_text, D_const, C_const, wk, sk, ranked_all_json
        )

    _audit = decision_json["audit"]
    _policy = decision_json["policy"]


    decision_json["audit"]["weather_key"] = wk
    decision_json["audit"]["scene_key"]   = sk

    llm_choice     = _policy["Final model"]
    shortlist_rows = _audit.get("shortlist_detail", [])
    R_llm       = float(_audit["R"])
    C_min_llm   = float(_audit["C_min"])
    C_max_llm   = float(_audit["C_max"])
    L_budget_llm = float(_audit["L_budget_ms"])

    rl_state = {
        "weather": wk,
        "scene": sk,
        "D": D_const,
        "C": C_const,
        "R": R_llm,
        "C_min": C_min_llm,
        "C_max": C_max_llm,
        "L_budget": L_budget_llm,
        "shortlist": shortlist_rows
    }

    if reuse_active_model:
        final_after_rl = llm_choice
        rl_decision = {"final_model": final_after_rl, "chosen_q": None, "signature": _signature_for_state(rl_state)}
        print("\n[RL DECISION] Adaptive Duration: reusing previous decision, RL selection skipped this frame.")
    else:
        rl_decision = rl_policy_select(rl_state, llm_choice)
        final_after_rl = rl_decision["final_model"]

    print("\n[RL DECISION] Before reward update:")
    print(f"  LLM final suggestion : {llm_choice}")
    print(f"  RL_Q-chosen model    : {final_after_rl}")
    print(f"  RL signature         : {rl_decision['signature']}")
    if rl_decision["chosen_q"] is not None:
        print(f"  RL chosen_q          : {rl_decision['chosen_q']:.4f}")

    decision_json["policy"]["Final model"] = final_after_rl

    candidate_models_llm = [row.get("model") for row in shortlist_rows]
    fallback_order = [final_after_rl] + [m for m in candidate_models_llm if m != final_after_rl]
    exec_result = DR.run_with_fallback(fallback_order, args.image)

    if exec_result is not None:
        if exec_result.model != final_after_rl:
            print(f"[EXECUTION] '{final_after_rl}' unavailable; executed "
                  f"'{exec_result.model}' instead.")
        executed_model = exec_result.model
        observed_confidence = exec_result.confidence
        observed_latency_ms = exec_result.latency_ms
        observed_reward, observed_penalty = compute_reward(
            observed_confidence, observed_latency_ms, L_budget_llm
        )
        decision_json["policy"]["Final model"] = executed_model
        rl_update_after_execution(
            rl_state, executed_model, observed_reward, observed_penalty
        )

        learned_confidence = WDB.record_confidence_observation(
            executed_model, wk, sk, observed_confidence
        )
        print(f"[LEARNED CONFIDENCE] {executed_model} @ {wk}/{sk}: "
              f"{learned_confidence:.4f} (this observation: {observed_confidence:.4f})")

        image_stem = os.path.splitext(os.path.basename(args.image))[0]
        annotated_path = os.path.join(
            args.frames_dir, f"{image_stem}_{executed_model.replace(' ', '_')}.jpg"
        )
        DR.save_annotated_image(
            args.image, exec_result.detections, annotated_path,
            title=f"{executed_model} | conf={observed_confidence:.2f} | {observed_latency_ms:.0f}ms",
        )
        print(f"[EXECUTION] Saved annotated output frame -> {annotated_path}")
    else:
        executed_model = None
        observed_confidence = None
        observed_latency_ms = None
        observed_reward, observed_penalty = None, None
        learned_confidence = None
        annotated_path = None
        print("[EXECUTION] No candidate detector could be executed; "
              "RL memory not updated for this frame.")

    extra_note = (
        f" RL memory suggests deploying {final_after_rl} in this "
        f"{wk}/{sk} scene (D≈{D_const:.3f}, C≈{C_const:.3f})."
    )
    if executed_model is not None:
        extra_note += (
            f" Executed {executed_model}: confidence={observed_confidence:.3f}, "
            f"latency={observed_latency_ms:.1f}ms, reward={observed_reward:.2f}."
        )
    if "reasoning" in decision_json and isinstance(decision_json["reasoning"], str):
        decision_json["reasoning"] += extra_note
    else:
        decision_json["reasoning"] = extra_note

    decision_json["policy"]["_rl_signature"] = rl_decision["signature"]
    # pick for scene_rate_metrics.py's Hallucination Correction Rate
    decision_json["policy"]["_llm_choice"] = llm_choice
    decision_json["policy"]["_rl_choice"] = final_after_rl
    decision_json["policy"]["_execution"] = {
        "model": executed_model,
        "confidence": observed_confidence,
        "latency_ms": observed_latency_ms,
        "reward": observed_reward,
        "latency_penalty": observed_penalty,
        "num_detections": exec_result.num_detections if exec_result else None,
        "backend": exec_result.backend if exec_result else None,
        "device": exec_result.device if exec_result else None,
        "learned_confidence_profile": learned_confidence,
        "annotated_frame": annotated_path,
    }

    sig_now = _signature_for_state(rl_state)
    key_str_now = json.dumps({"sig": sig_now, "model": final_after_rl})

    print("\n[RL DEBUG] Current frame summary:")
    print(f"  weather/scene: {rl_state['weather']} / {rl_state['scene']}")
    print(f"  D={rl_state['D']:.3f}, C={rl_state['C']:.3f}, R={rl_state['R']:.3f}, "
          f"C_min={rl_state['C_min']:.3f}, C_max={rl_state['C_max']:.3f}, "
          f"L_budget={rl_state['L_budget']:.1f} ms")
    print(f"  shortlist = {[row['model'] for row in rl_state['shortlist']]}")
    print(f"  LLM chose = {llm_choice}")
    print(f"  RL final  = {final_after_rl}")
    print(f"  executed  = {executed_model}")
    print(f"  confidence/latency = {observed_confidence}, {observed_latency_ms}")
    print(f"  reward/penalty given = {observed_reward}, {observed_penalty}")
    print(f"  Q[{final_after_rl}] for this signature is now {RL_Q.get(key_str_now, 0.0):.4f}")

    print("\n[RL DEBUG] Full RL_Q memory:")
    for k_str, qv in RL_Q.items():
        try:
            parsed = json.loads(k_str)
        except Exception:
            parsed = {"sig": "???", "model": "???"}
        print(f"  sig={parsed['sig']}, model={parsed['model']}, Q={qv:.4f}")
    print()


    decision_json["policy"]["_rl_debug_Q"] = {
        json.dumps({"sig": rl_decision["signature"], "model": row["model"]}):
            RL_Q.get(json.dumps({"sig": rl_decision["signature"], "model": row["model"]}), 0.0)
        for row in shortlist_rows
    }

    if reuse_active_model:
        _next_remaining = _remaining_duration - 1
        _duration_rl_state = _stored_rl_state
        _duration_used_mode = _stored_rl_state.get("used_mode", "REUSED")
    else:
        _c_hat_now = (C_const / C_max_seen) if C_max_seen > 0 else 0.0
        _new_tau = DT.compute_duration(
            similarity_now, _c_hat_now,
            tau_min=args.duration_min_frames, tau_max=args.duration_max_frames,
        )
        _next_remaining = _new_tau - 1
        _duration_rl_state = dict(rl_state)
        _duration_rl_state["used_mode"] = _audit.get("used_mode", "IN_BAND")
        _duration_used_mode = _duration_rl_state["used_mode"]
        print(f"[DURATION] Reevaluated -> new tau_t={_new_tau} frame(s) "
              f"(S_hat_t={similarity_now:.4f}, C_hat_t={_c_hat_now:.4f})")

    DT.save_state({
        "feature_vector": feature_vector_now,
        "active_model": final_after_rl,
        "remaining_duration": _next_remaining,
        "C_max_seen": C_max_seen,
        "rl_state": _duration_rl_state,
        "used_mode": _duration_used_mode,
    })

    out = {
        "meta": {
            "image": os.path.basename(args.image),
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": {
                "reused_previous_decision": reuse_active_model,
                "similarity_S_hat_t": similarity_now,
                "remaining_duration_after_this_frame": _next_remaining,
                "C_max_seen": C_max_seen,
            },
        },
        "step1_scene_context_extraction": {
            "raw": desc_text,
            "json": desc_json,
            "tokens": desc_tokens,
            "time_sec": desc_time
        },
        "step2_scene_context_scoring": {
            "raw": analysis_text,
            "json": analysis_json,
            "tokens": analysis_tokens,
            "time_sec": analysis_time
        },
        "step3_scene_aware_decision": {
            "raw": decision_text,
            "json": decision_json,
            "tokens": decision_tokens,
            "time_sec": decision_time
        }
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(rl_memory_path, "w") as fmem:
        json.dump(RL_Q, fmem)

    WDB.save_confidence_overrides()

    print(f"Saved CoT results -> {args.out}\n")


if __name__ == "__main__":
    main()