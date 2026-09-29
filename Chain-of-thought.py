import argparse
import json
import math
import os
import re
import time
import yaml
from typing import Any, Dict, List, Optional

from vlm import ModelHandler
import analysis_settings as S
import winners_model_DB as WDB
import detector_runner as DR
import duration_tracker as DT


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description="SAVER CoT: Scene Context Extraction + Scene Context Scoring + "
                     "Scene-aware Decision (Qwen, LightEMMA-style chaining), no RL"
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


def strip_md_fences(text: str) -> str:
    return re.sub(r"```[a-zA-Z]*\n?|```", "", text).strip()


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:

    raw = strip_md_fences(text)

    raw = re.sub(r'//.*', '', raw)

    raw = re.sub(r'/\*.*?\*/', '', raw, flags=re.S)
    raw = re.sub(r',\s*([}\]])', r'\1', raw)

    def eval_expr(match):
        expr = match.group(0)

        if re.search(r'[^0-9\.\+\-\*\s]', expr):
            return expr
        try:
            val = eval(expr, {"__builtins__": {}}, {})
            return f"{float(val):.6f}"
        except Exception:
            return expr

    math_pattern = re.compile(
        r'(\d+(?:\.\d+)?(?:\s*[\+\*]\s*\d+(?:\.\d+)?)+)'
    )
    raw = math_pattern.sub(eval_expr, raw)

    start = raw.find('{')
    if start == -1:
        return None
    depth, end = 0, None
    for i, ch in enumerate(raw[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None

    obj_txt = raw[start:end]

    obj_txt = re.sub(r'<\s*(float|int|str)\s*>', '0', obj_txt)

    try:
        obj = json.loads(obj_txt)
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        print("[DEBUG] JSON decode failed:", e)
        print("[DEBUG] raw (first 600 chars):")
        print(obj_txt[:600])
        return None


def _expected_shortlist_and_final(ranked_all: List[Dict[str, Any]], L_budget_ms: float, C_min: float):
    """The deterministic version of sub-steps (3)+(4) from
    build_shortlist_and_final_decision_prompt's own rules -- a mechanical
    filter/sort over ranked_all, not a judgment call, so it can be (and
    should be) recomputed in Python and used to catch the LLM getting the
    filtering wrong (e.g. returning ranked_all's highest-mAP row regardless
    of latency and mislabeling it "IN_BAND" instead of "FALLBACK").
    """
    feasible = [r for r in ranked_all
                if r["latency"] <= L_budget_ms and (r["confidence"] is None or r["confidence"] >= C_min)]
    if feasible:
        max_map = max(r["mAP"] for r in feasible)
        tied = [r for r in feasible if math.isclose(r["mAP"], max_map, rel_tol=1e-9, abs_tol=1e-9)]
        return "IN_BAND", feasible[:3], min(tied, key=lambda r: r["latency"])
    if ranked_all:
        fastest = min(ranked_all, key=lambda r: r["latency"])
        return "FALLBACK", [fastest], fastest
    return "FALLBACK", [], None


def _verify_shortlist_final(shortlist_json: Dict[str, Any], ranked_all: List[Dict[str, Any]],
                             L_budget_ms: float, C_min: float):
    """Recheck the LLM's own (3)+(4) Shortlist/Final-Decision answer against
    the real winner-model table. Returns (ok, expected_mode,
    expected_shortlist, expected_final_row); ok is False on any mismatch
    (including a malformed shortlist_json)."""
    expected_mode, expected_shortlist, expected_final = _expected_shortlist_and_final(
        ranked_all, L_budget_ms, C_min
    )
    try:
        used_mode = shortlist_json["used_mode"]
        shortlist_detail = shortlist_json["shortlist_detail"]
        policy = shortlist_json["policy"]
        ok = (
            used_mode == expected_mode
            and policy.get("K") == len(shortlist_detail)
            and [r["model"] for r in shortlist_detail] == [r["model"] for r in expected_shortlist]
            and expected_final is not None
            and policy.get("Final model") == expected_final["model"]
        )
    except (KeyError, TypeError, ValueError):
        ok = False
    return ok, expected_mode, expected_shortlist, expected_final


def run_risk_assessment(model, args, D_const, C_const):
    """Step 3, sub-step (1) Risk Assessment / Risk Factor Aggregation
    (paper Sec. III.B.3): computes R and nothing else. This is
    deliberately the smallest possible call.
    Returns (R, risk_json, risk_text, tokens, time).
    """
    risk_prompt = build_risk_assessment_prompt(D_const, C_const)

    print("[Step 3] Scene-aware Decision -- (1) Risk Assessment …\n")
    risk_text, risk_tokens, risk_time = model.get_response(
        prompt=risk_prompt,
        image_path=args.image,
        max_tokens=400
    )
    print(risk_text, "\n")
    print("   Tokens:", risk_tokens, "  Time:", f"{risk_time:.2f}s\n")

    risk_json = try_parse_json(risk_text)
    if not risk_json or "R" not in risk_json:
        raise RuntimeError(
            "Step 3 Risk Assessment JSON parse failed or missing 'R'; "
            "cannot proceed without the model's own risk-factor calculation."
        )

    R = float(risk_json["R"])
    return R, risk_json, risk_text, risk_tokens, risk_time


def build_risk_assessment_prompt(D_const: float, C_const: float) -> str:
    """The exact prompt text run_risk_assessment sends -- factored out so
    fine-tuning.py can regenerate the identical prompt for a historical
    (D, C) pair when mining training data, without a hand-duplicated copy
    that could silently drift from the real prompt."""
    return f"""
    You are an autonomous driving calculation system.
    Return ONLY the final JSON (no markdown, no extra text).

    Inputs (authoritative; already computed in a prior step -- use exactly,
    do not recompute D or C themselves):
    - D = {D_const:.6f}
    - C = {C_const:.6f}

    Compute the risk factor:
        C_norm = C / 20.0
        R = 0.60 * D + 0.40 * C_norm
    Round C_norm and R to 6 decimals.

    Self-check (MANDATORY): R must equal 0.60*D + 0.40*C_norm exactly using
    the D and C_norm you wrote below. If it doesn't, recompute and fix it
    before output.

    Return ONLY this JSON:
    {{
    "terms": {{
        "C_norm": <float>,
        "risk_D_term": <float, = 0.60 * D>,
        "risk_C_term": <float, = 0.40 * C_norm>
    }},
    "R": <float, = terms.risk_D_term + terms.risk_C_term>,
    "calculation": "R = 0.60*D + 0.40*C_norm = <terms.risk_D_term> + <terms.risk_C_term> = <R>"
    }}
    """.strip()


def run_confidence_policy(model, args, R, wk):
    """Step 3, sub-step (2)(1) Confidence policy (paper Sec. III.B.3):
    computes C_min, C_max ONLY. R is passed in as a FIXED, GIVEN constant
    (computed by run_risk_assessment).
    Returns (C_min, C_max, cp_json, cp_text, tokens, time).
    """
    cp_prompt = build_confidence_policy_prompt(R, wk)

    print("[Step 3] Scene-aware Decision -- (2)(1) Confidence Policy …\n")
    cp_text, cp_tokens, cp_time = model.get_response(
        prompt=cp_prompt,
        image_path=args.image,
        max_tokens=500
    )
    print(cp_text, "\n")
    print("   Tokens:", cp_tokens, "  Time:", f"{cp_time:.2f}s\n")

    cp_json = try_parse_json(cp_text)
    _required = ["C_min", "C_max"]
    if not cp_json or any(k not in cp_json for k in _required):
        raise RuntimeError(
            "Step 3 Confidence Policy JSON parse failed or missing required "
            "field(s); cannot proceed without the model's own "
            "confidence-policy calculation."
        )

    C_min = float(cp_json["C_min"])
    C_max = float(cp_json["C_max"])
    return C_min, C_max, cp_json, cp_text, cp_tokens, cp_time


def build_confidence_policy_prompt(R: float, wk: str) -> str:
    """The exact prompt text run_confidence_policy sends -- see
    build_risk_assessment_prompt's docstring for why this is factored out."""
    return f"""
    You are an autonomous driving calculation system.
    Return ONLY the final JSON (no markdown, no extra text).

    Inputs (authoritative; already computed in a prior step -- use R
    exactly as given, do not recompute it):
    - R = {R:.6f}
    - weather_key = "{wk}"

    Confidence policy (STRICT; compute from an explicit picked block):
    - Pick (base_low, base_high) by matching weather_key against this table
      (exactly one row applies):
            weather_key == "Sunny"          -> (base_low, base_high) = (0.70, 0.75)
            weather_key in ("Rainy","Snow") -> (base_low, base_high) = (0.65, 0.75)
            otherwise (Foggy, Sand, ...)    -> (base_low, base_high) = (0.70, 0.75)
    - Output a "picked" field echoing the exact row you matched:
        "picked": {{"base_low": <base_low>, "base_high": <base_high>}}
    - Compute term-by-term, using R EXACTLY AS GIVEN ABOVE ({R:.6f}) both times:
        conf_min_term = 0.04 * {R:.6f}
        conf_max_term = 0.03 * {R:.6f}
        C_min = picked.base_low  + conf_min_term
        C_max = picked.base_high + conf_max_term
      NOTE on the paper's cap: the full formula is C_min = min(0.80, ...) and
      C_max = min(0.80, ...). That cap can be IGNORED here and NEVER equals
      the answer: base_low is at most 0.70 and R is at most 1.0, so
      C_min is at most 0.70 + 0.04 = 0.74, and C_max is at most
      0.75 + 0.03 = 0.78 -- both always strictly below 0.80. If your C_min
      or C_max comes out as exactly 0.80, that is wrong -- go back and
      actually add conf_min_term / conf_max_term.

    Self-check (MANDATORY):
    - Check C_min == picked.base_low  + terms.conf_min_term (no cap applied -- see note above).
    - Check C_max == picked.base_high + terms.conf_max_term (no cap applied -- see note above).
    - If ANY mismatch occurs, recompute and FIX the numbers so equalities
      hold EXACTLY before output.

    Return ONLY this JSON:
    {{
    "picked": {{"base_low": <float>, "base_high": <float>}},
    "terms": {{"conf_min_term": <float>, "conf_max_term": <float>}},
    "C_min": <float>,
    "C_max": <float>,
    "calculation": "C_min = base_low + 0.04*R = <picked.base_low> + <terms.conf_min_term> = <C_min>; C_max = base_high + 0.03*R = <picked.base_high> + <terms.conf_max_term> = <C_max>"
    }}
    """.strip()


def run_latency_budget_policy(model, args, R, sk):
    """Step 3, sub-step (2)(2) Latency budget policy (paper Sec. III.B.3):
    computes L_budget_ms ONLY.
    Returns (L_budget_ms, lb_json, lb_text, tokens, time).
    """
    lb_prompt = build_latency_budget_prompt(R, sk)

    print("[Step 3] Scene-aware Decision -- (2)(2) Latency Budget Policy …\n")
    lb_text, lb_tokens, lb_time = model.get_response(
        prompt=lb_prompt,
        image_path=args.image,
        max_tokens=400
    )
    print(lb_text, "\n")
    print("   Tokens:", lb_tokens, "  Time:", f"{lb_time:.2f}s\n")

    lb_json = try_parse_json(lb_text)
    if not lb_json or "L_budget_ms" not in lb_json:
        raise RuntimeError(
            "Step 3 Latency Budget Policy JSON parse failed or missing "
            "'L_budget_ms'; cannot proceed without the model's own "
            "latency-budget calculation."
        )

    L_budget_ms = float(lb_json["L_budget_ms"])
    return L_budget_ms, lb_json, lb_text, lb_tokens, lb_time


def build_latency_budget_prompt(R: float, sk: str) -> str:
    """The exact prompt text run_latency_budget_policy sends -- see
    build_risk_assessment_prompt's docstring for why this is factored out."""
    return f"""
    You are an autonomous driving calculation system.
    Return ONLY the final JSON (no markdown, no extra text).

    Inputs (authoritative; already computed in a prior step -- use R
    exactly as given, do not recompute it. This R has nothing to do with
    confidence thresholds -- do not substitute any other number for it):
    - R = {R:.6f}
    - scene_key = "{sk}"

    Latency budget (STRICT; compute from an explicit picked block):
    - Pick (L_min, L_max) by matching scene_key against this table (exactly
      one row applies):
            scene_key == "Downtown" -> (L_min, L_max) = (200.0, 500.0)
            scene_key == "Highway"  -> (L_min, L_max) = (50.0, 150.0)
    - Output a "picked" field echoing the exact row you matched:
        "picked": {{"L_min": <L_min>, "L_max": <L_max>}}
    - Compute term-by-term, using R EXACTLY AS GIVEN ABOVE ({R:.6f}):
        latency_range_term = (picked.L_max - picked.L_min) * {R:.6f}
        L_budget_ms = picked.L_min + latency_range_term
      L_budget_ms is L_min PLUS that product -- it is NEITHER the product
      alone NOR (L_max - L_min) alone. If your L_budget_ms comes out
      exactly equal to (L_max - L_min), you forgot to multiply by R AND
      forgot to add L_min -- that is wrong, redo it.

    Self-check (MANDATORY):
    - Check L_budget_ms == picked.L_min + terms.latency_range_term.
    - Check terms.latency_range_term == (picked.L_max - picked.L_min) * {R:.6f} exactly --
      if you used any number other than {R:.6f} for R here, that is wrong, fix it.
    - If ANY mismatch occurs, recompute and FIX the numbers so equalities
      hold EXACTLY before output.

    Return ONLY this JSON:
    {{
    "picked": {{"L_min": <float>, "L_max": <float>}},
    "terms": {{"latency_range_term": <float>}},
    "L_budget_ms": <float>,
    "calculation": "L_budget_ms = L_min + (L_max-L_min)*R = <picked.L_min> + <terms.latency_range_term> = <L_budget_ms>"
    }}
    """.strip()


def run_shortlist_and_final_decision(model, args, desc_text, analysis_text, D_const, C_const, wk, sk,
                                      R, C_min, C_max, L_budget_ms, ranked_all_json):
    """Step 3, sub-steps (3) Shortlist Candidates and (4) Final Decision
    (paper Sec. III.B.3): shortlisting and final-model selection ONLY.
    R, C_min, C_max, L_budget_ms are passed in already computed by this
    module's run_risk_assessment / run_confidence_policy /
    run_latency_budget_policy sub-steps -- this call does no
    risk/confidence/latency arithmetic at all, only
    filtering/ranking ranked_all_json against those fixed numbers, which
    is a task LLMs handle far more reliably than the multi-step arithmetic. 
    Even so, the (3)+(4) filter is purely
    mechanical, so its output is cross-checked against
    a deterministic recomputation from ranked_all_json below and corrected
    if the model got the latency/confidence filtering wrong -- e.g.
    returning the single highest-mAP row regardless of latency and
    mislabeling it "IN_BAND" instead of correctly filtering to the
    latency-feasible subset.
    Returns (decision_json, decision_text, decision_tokens, decision_time).
    """
    shortlist_prompt = build_shortlist_and_final_decision_prompt(
        desc_text, analysis_text, D_const, C_const, wk, sk, R, C_min, C_max, L_budget_ms, ranked_all_json
    )

    print("[Step 3] Scene-aware Decision -- (3)+(4) Shortlist Candidates & Final Decision …\n")
    decision_text, decision_tokens, decision_time = model.get_response(
        prompt=shortlist_prompt,
        image_path=args.image,
        max_tokens=1200
    )
    print(decision_text, "\n")
    print("   Tokens:", decision_tokens, "  Time:", f"{decision_time:.2f}s\n")

    shortlist_json = try_parse_json(decision_text)
    if not shortlist_json or "policy" not in shortlist_json:
        raise RuntimeError("Step 3 Shortlist/Final-Decision JSON parse failed; missing 'policy'.")

    _policy = shortlist_json["policy"]
    _required = ["used_mode", "shortlist_detail"]
    _missing = [k for k in _required if k not in shortlist_json] + (["policy.Final model"] if "Final model" not in _policy else [])
    if _missing:
        raise RuntimeError(
            f"Step 3 Shortlist/Final-Decision output is missing required "
            f"field(s) {_missing}; cannot proceed without the model's own "
            f"shortlist/final-model selection."
        )

    ranked_all = json.loads(ranked_all_json)

    _reported_feasible = shortlist_json.get("feasible")
    if isinstance(_reported_feasible, list):
        _real_feasible_models = [
            r["model"] for r in ranked_all
            if r["latency"] <= L_budget_ms and (r["confidence"] is None or r["confidence"] >= C_min)
        ]
        if _reported_feasible != _real_feasible_models:
            print(
                "[WARN] Step 3's own 'feasible' field does not match the real "
                "latency/confidence-filtered set -- it likely skipped the "
                f"per-row latency check.\n       LLM said feasible={_reported_feasible!r}\n"
                f"       Correct feasible={_real_feasible_models!r}\n"
            )

    ok, expected_mode, expected_shortlist, expected_final = _verify_shortlist_final(
        shortlist_json, ranked_all, L_budget_ms, C_min
    )
    if not ok:
        got_final = shortlist_json.get("policy", {}).get("Final model")
        print(
            "[WARN] Step 3 (3)+(4) Shortlist/Final-Decision failed verification "
            "against the real latency/confidence table -- overriding with the "
            "deterministically correct shortlist.\n"
            f"       LLM said: used_mode={shortlist_json.get('used_mode')!r} "
            f"K={shortlist_json.get('policy', {}).get('K')} "
            f"Final model={got_final!r}\n"
            f"       Correct:  used_mode={expected_mode!r} "
            f"K={len(expected_shortlist)} "
            f"Final model={(expected_final or {}).get('model')!r} "
            f"(L_budget_ms={L_budget_ms:.2f}, C_min={C_min:.4f})\n"
        )
        shortlist_json = {
            "used_mode": expected_mode,
            "shortlist_detail": expected_shortlist,
            "policy": {
                "K": len(expected_shortlist),
                "candidates": [r["model"] for r in expected_shortlist],
                "Final model": (expected_final or {}).get("model"),
            },
            "reasoning": (
                "[auto-corrected: the model's own shortlist/final-decision did "
                "not match the real latency/confidence-filtered table, so it "
                "was replaced with the deterministic result] "
                + shortlist_json.get("reasoning", "")
            ).strip(),
        }

    return shortlist_json, decision_text, decision_tokens, decision_time


def build_shortlist_and_final_decision_prompt(desc_text, analysis_text, D_const, C_const, wk, sk,
                                               R, C_min, C_max, L_budget_ms, ranked_all_json) -> str:
    """The exact prompt text run_shortlist_and_final_decision sends -- see
    build_risk_assessment_prompt's docstring for why this is factored out."""
    return f"""
    You are an autonomous driving decision planner.
    RETURN ONLY THE JSON SHAPE SHOWN AT THE END (no markdown fences, no prose outside the JSON).

    Inputs:
    - Scene Description: {strip_md_fences(desc_text)}
    - Scene Analysis:    {strip_md_fences(analysis_text)}
    - Difficulty D (from Step 2; already used to derive the numbers below): {D_const:.6f}
    - Complexity C (from Step 2; already used to derive the numbers below): {C_const:.6f}
    - weather_key = "{wk}"
    - scene_key   = "{sk}"

    The following FOUR numbers were already computed by this Step's own
    prior sub-steps, (1) Risk Assessment and (2) Dynamic Policy, from D and
    C -- they are FIXED INPUTS to this sub-step. Do NOT recompute, adjust,
    or second-guess them; use them exactly as given:
    - R           = {R:.6f}
    - C_min       = {C_min:.6f}
    - C_max       = {C_max:.6f}
    - L_budget_ms = {L_budget_ms:.6f}

    Authoritative Winner Model Database slice for this weather_key/scene_key
    (DO NOT re-sort, drop, or add rows -- this table is fixed ground truth):
    ranked_all (sorted by mAP desc, tie -> latency asc; "confidence" is a real
    offline-profiled or learned-online value where present, or null where
    none exists yet -- null means "no data for this row, do not apply the
    confidence test to it"):
    {ranked_all_json}

    (3) Shortlist Candidates -- using L_budget_ms and C_min given above:
        FIRST, check EVERY row in ranked_all, one at a time, against BOTH
        conditions below. Do this even for the row with the highest mAP --
        a high-mAP row that fails the latency check is NOT feasible, no
        matter how good its mAP is:
            latency <= L_budget_ms  AND  (confidence is null OR confidence >= C_min)
        Collect every row that passes BOTH conditions into "feasible",
        keeping ranked_all's relative order, and report it explicitly in
        the "feasible" field of your output (model names only) -- this is
        proof you actually checked each row's latency, not just picked the
        top of ranked_all.

        feasible is non-empty in the large majority of real scenes, since
        at least one fast, lower-mAP model in ranked_all almost always
        meets a real latency budget. Do NOT default to treating feasible
        as empty without having explicitly checked every row above.

        - Normal case (feasible is non-empty, after the explicit check above):
            used_mode = "IN_BAND"
            Shortlist = the first min(3, len(feasible)) rows of feasible, in order.
        - Rare exception (feasible is empty -- you checked every row above
          and NONE of them satisfy BOTH conditions):
            used_mode = "FALLBACK"
            Shortlist = the single FASTEST row in ranked_all (lowest
            latency), NOT the highest-mAP row -- when nothing meets the
            budget, minimizing latency is prioritized over accuracy for
            this frame; accuracy is only the tie-break once feasibility
            already holds (see the Normal case above).
        Shortlist may legally have length 1, 2, or 3.
        policy.K MUST equal len(Shortlist); policy.candidates MUST list ONLY
        the models actually in Shortlist, in order.

    (4) Final Decision:
        - If feasible is non-empty: among rows in feasible whose mAP equals
          the maximum mAP in feasible, FinalRow = the one with the lowest latency.
        - Else (FALLBACK): FinalRow = the single fastest row in ranked_all
          (the same row as Shortlist[0] in this case).
        You are NOT allowed to pick a lower-mAP model just because it is
        faster: e.g. if "Mask-RCNN" and "Deformable DETR (R50)" are tied at
        the max mAP in feasible, FinalRow is whichever of THOSE TWO has
        lower latency -- never a third, lower-mAP model just because it's
        faster than both.

    FINAL OUTPUT SHAPE (fill with the real numbers/rows YOU selected -- no placeholders, no extra keys):
    {{
    "feasible": ["MODEL_NAME", "..."],
    "used_mode": "<IN_BAND|FALLBACK>",
    "shortlist_detail": [
        {{"model": "MODEL_1", "latency": LAT_1, "mAP": MAP_1, "confidence": CONF_1_or_null}}
        // add MODEL_2 / MODEL_3 the same way if Shortlist has them
    ],
    "policy": {{
        "K": <len_of_shortlist>,
        "candidates": ["MODEL_1","MODEL_2 (if exists)","MODEL_3 (if exists)"],
        "Final model": "FINALROW_MODEL"
    }},
    "reasoning": "<2-3 sentences: name FinalRow.model, its mAP/latency, and whether it met the latency+confidence budget or required FALLBACK>"
    }}
    """.strip()


def run_scene_aware_decision(model, args, desc_text, analysis_text, D_const, C_const, wk, sk, ranked_all_json):
    """Step 3: Scene-aware Decision (paper Sec. III.B.3). The third module
    of SAVER's structured CoT, which selects the best model using four
    sub-steps: (1) Risk Assessment (Risk Factor Aggregation), (2) Dynamic
    Policy (confidence policy + latency budget policy), (3) Shortlist
    Candidates, (4) Final Decision.

    Implemented as FOUR focused LLM calls -- run_risk_assessment,
    run_confidence_policy, run_latency_budget_policy,
    run_shortlist_and_final_decision -- kept under this one entry
    point so users don't need to know about the split. Returns
    (decision_json, decision_text, decision_tokens, decision_time), with
    decision_json["audit"] carrying D/C/R/C_min/C_max/L_budget_ms/
    used_mode/shortlist_detail and decision_json["policy"] carrying
    K/candidates/"Final model" -- entirely the VLM's own CoT output
    across the four calls, with no Python-side computation or override
    of any of them.
    """
    R, risk_json, risk_text, risk_tokens, risk_time = \
        run_risk_assessment(model, args, D_const, C_const)

    C_min, C_max, cp_json, cp_text, cp_tokens, cp_time = \
        run_confidence_policy(model, args, R, wk)

    L_budget_ms, lb_json, lb_text, lb_tokens, lb_time = \
        run_latency_budget_policy(model, args, R, sk)

    shortlist_json, shortlist_text, shortlist_tokens, shortlist_time = \
        run_shortlist_and_final_decision(
            model, args, desc_text, analysis_text, D_const, C_const, wk, sk,
            R, C_min, C_max, L_budget_ms, ranked_all_json
        )

    decision_json = {
        "audit": {
            "D": D_const,
            "C": C_const,
            "weather_key": wk,
            "scene_key": sk,
            "picked": {**risk_json.get("terms", {}), **cp_json.get("picked", {}), **lb_json.get("picked", {})},
            "terms": {**risk_json.get("terms", {}), **cp_json.get("terms", {}), **lb_json.get("terms", {})},
            "R": R,
            "C_min": C_min,
            "C_max": C_max,
            "L_budget_ms": L_budget_ms,
            "used_mode": shortlist_json["used_mode"],
            "shortlist_detail": shortlist_json["shortlist_detail"],
        },
        "policy": shortlist_json["policy"],
        "reasoning": shortlist_json.get("reasoning", ""),
    }

    decision_text = (
        "--- Step 3 (1) Risk Assessment ---\n" + risk_text +
        "\n\n--- Step 3 (2)(1) Confidence Policy ---\n" + cp_text +
        "\n\n--- Step 3 (2)(2) Latency Budget Policy ---\n" + lb_text +
        "\n\n--- Step 3 (3)+(4) Shortlist Candidates & Final Decision ---\n" + shortlist_text
    )
    _tok = lambda t, k: t.get(k, 0) if isinstance(t, dict) else 0
    decision_tokens = {
        "input": _tok(risk_tokens, "input") + _tok(cp_tokens, "input") + _tok(lb_tokens, "input") + _tok(shortlist_tokens, "input"),
        "output": _tok(risk_tokens, "output") + _tok(cp_tokens, "output") + _tok(lb_tokens, "output") + _tok(shortlist_tokens, "output"),
    }
    decision_time = risk_time + cp_time + lb_time + shortlist_time

    return decision_json, decision_text, decision_tokens, decision_time


def run_scene_context_extraction(model, args):
    """Step 1: Scene Context Extraction (paper Sec. III.B.1). The first
    module of SAVER's structured CoT: extracts a structured, machine-
    readable representation of the driving scene directly from the input
    image, via (1) Environment Description (weather, time, road, lane) and
    (2) Critical Object Identification (dynamic/static critical objects).
    This scene state is the shared input Step 2 (Scene Context Scoring)
    and Step 3 (Scene-aware Decision) both build on.
    Returns (desc_json, desc_text, desc_tokens, desc_time).
    """
    scene_description_prompt = build_scene_context_extraction_prompt()

    print("[Step 1] Scene Context Extraction …\n")
    desc_text, desc_tokens, desc_time = model.get_response(
        prompt=scene_description_prompt,
        image_path=args.image
    )

    print(desc_text, "\n")
    print("   Tokens:", desc_tokens, "  Time:", f"{desc_time:.2f}s\n")

    desc_json = try_parse_json(desc_text)
    return desc_json, desc_text, desc_tokens, desc_time


def build_scene_context_extraction_prompt() -> str:
    """The exact prompt text run_scene_context_extraction sends -- see
    build_risk_assessment_prompt's docstring for why this is factored out.
    Unlike the other five, this prompt is constant (no per-frame values are
    interpolated into it); it's still factored out for the same reason:
    Step 1's SFT examples (fine-tuning.py) must be built from the exact
    prompt the live pipeline sends, not a hand-copied one."""
    return """
        You are an autonomous driving perception system.
        Analyze the front-view driving image and output ONLY this JSON:

        {
        "environment": {
            "weather": one of ["sandstorm","sunny","snow","rain","fog","clear","cloudy","overcast","mixed"],
            "time":    one of ["day","dusk","night","dawn"],
            "road":    short string (e.g., "highway","urban intersection","residential street","rural road","roundabout","tunnel","bridge","ramp","busy downtown"),
            "lane":    short string describing lane type/condition (e.g.,
                        "two-lane", "multi-lane", "one-way",
                        "lane blocked", "lane closed", "construction zone",
                        "merge", "split", "exit ramp", "entry ramp", "lane drop",
                        "weaving section", "diverging lane",
                        "no marking", "faded markings", "poorly marked",
                        "wet surface", "icy surface", "temporary markings",
                        "sharp curve", "steep incline", "tunnel lane", "bridge lane",
                        "default")
        },
        "critical_objects": [
            // ZERO OR MORE items.
            //
            // IMPORTANT:
            // "critical_objects" MUST include ONLY objects that force an immediate driving action
            // (brake NOW, steer NOW, or risk collision NOW).
            //
            // Examples of TRUE critical objects:
            // - a pedestrian / runner / animal crossing in front of the ego vehicle
            // - a vehicle merging / cutting in / braking hard directly in front of ego
            // - debris or obstacle actually blocking the ego lane
            // - something severely blocking the driver's forward view (snow load, fallen tree, etc.)
            //
            // The following are NOT critical by themselves:
            // - normal traffic ahead moving in the same lane and same direction at normal distance
            // - distant vehicles in other lanes
            // - vehicles simply being followed
            //
            // Before finalizing this list, scan the WHOLE image twice, once
            // for each category below -- STATIC hazards are just as easy to
            // miss as dynamic ones because they don't move, but they are
            // just as critical when they block the lane or the driver's view:
            //   (1) DYNAMIC: pedestrians, cyclists, motorcyclists, animals,
            //       any vehicle merging/cutting in/braking hard near ego.
            //   (2) STATIC/ENVIRONMENTAL: snow or ice piles, debris,
            //       potholes, fallen trees/branches, construction equipment,
            //       stalled or accident vehicles blocking the lane -- check
            //       the road surface and shoulders directly ahead, not just
            //       oncoming/moving traffic.
            // Only after checking BOTH categories, decide if any qualify as
            // truly critical per the definition above.
            //
            // If there are NO true critical objects, set:
            // "critical_objects": []

            {
            "class": one of [
                "pedestrian", "construction worker", "wheelchair user", "crossing guard", "runner", "scooter rider",
                "motorcyclist", "cyclist", "police car", "ambulance", "fire truck", "emergency vehicle", "Construction zone car",
                "Train", "school bus", "taxis", "van", "tow vehicle", "bicycle", "scooter",
                "e scooter", "motorcycle", "delivery robots", "cattle", "sheep", "horse", "deer",
                "dog", "cat", "wild animal", "animal other", "crosswalk sign", "traffic light",
                "stop sign", "yield sign", "speed bump", "direction arrow", "construction zone sign", "road marking faded",
                "cone", "barrier", "barrier gate", "crossing gate", "trash bin", "construction equipment", "garbage bag", "snow pile",
                "pothole", "fallen tree", "Parked car", "parked truck", "parked bus", "parked motorcycle",
                "accident car", "accident truck", "accident bus", "accident motorcycle",
                "Trees", "plastic bags", "buildings", "fences"
            ],
            "bbox": [x1,y1,x2,y2],        // approximate pixel coordinates if visible
            "influence": short phrase describing its current role or risk
                        (e.g.,
                        "crossing ahead suddenly","snow load occluding view", "hazard to visibility ahead",
                        "near the side of the road", "blocking the ego vehicle", "crossing the road", "turning left blocking the ego vehicle",
                        "approaching the ego vehicle", "cutting in", "occluding view of the ego vehicle", "approaching towards the vehicle",
                        "falling debris risk", "slippery surface risk", "animal crossing risk","parked on roadside", "accident ahead", "construction zone hazard",
                        "obstructing driver's view", "debris on road", "potential collision risk")
            }
        ]
        }

        Rules:
        - Return ONLY valid JSON (no markdown fences, no extra text).
        - Include ONLY objects that are critical to immediate driving decisions (prefer 0–6 items).
        - If there are NO critical objects, set "critical_objects": [].
        """.strip()


def run_scene_context_scoring(model, args, desc_text):
    """Step 2: Scene Context Scoring (paper Sec. III.B.2). The second
    module of SAVER's structured CoT: converts Step 1's structured scene
    state into machine-readable difficulty and complexity scores -- (1)
    Difficulty Score Estimation D(s) from environmental attributes and (2)
    Complexity Score Estimation C(s) from critical objects. (The module's
    third sub-step, (3) Duration/Adaptive Duration, needs no LLM call and
    is implemented separately in duration_tracker.py; see main().)
    Returns (analysis_json, analysis_text, analysis_tokens, analysis_time).
    """
    scene_analysis_prompt = build_scene_context_scoring_prompt(desc_text)

    print("[Step 2] Scene Context Scoring …\n")
    analysis_text, analysis_tokens, analysis_time = model.get_response(
        prompt=scene_analysis_prompt,
        image_path=args.image,
        max_tokens=1500
    )

    print(analysis_text, "\n")
    print("   Tokens:", analysis_tokens, "  Time:", f"{analysis_time:.2f}s\n")

    analysis_json = try_parse_json(analysis_text)
    if analysis_json is None:
        raise RuntimeError("Step 2 Scene Context Scoring JSON parse failed; cannot proceed to Step 3.")

    return analysis_json, analysis_text, analysis_tokens, analysis_time


def build_scene_context_scoring_prompt(desc_text: str) -> str:
    """The exact prompt text run_scene_context_scoring sends -- see
    build_risk_assessment_prompt's docstring for why this is factored out."""
    cfg_for_prompt = S.as_prompt_dicts()

    return f"""
    You are an autonomous driving calculation and reasoning system.
    Return ONLY the final JSON (no markdown, no extra text).

    Inputs
    - Weights (authoritative; DO NOT alter): {json.dumps(cfg_for_prompt["weights"])}
    - weather_scores: {json.dumps(cfg_for_prompt["weather_scores"])}
    - time_scores:    {json.dumps(cfg_for_prompt["time_scores"])}
    - road_scores:    {json.dumps(cfg_for_prompt["road_scores"])}
    - lane_rules: {json.dumps(cfg_for_prompt["lane_rules"])}
    - Complexity constants: alpha={cfg_for_prompt["alpha"]}, beta={cfg_for_prompt["beta"]}
    - Priority table (use these exact keys; do NOT invent new weights): {json.dumps(cfg_for_prompt["priority"])}
    - Class normalization rules (map to the exact priority key):
    * "pedestrian crossing" -> "pedestrian"
    * "people" -> "pedestrian"
    * "person" -> "pedestrian"
    * "bicyclist"/"bike rider" -> "cyclist"
    * "motorbike"/"moto" -> "motorcycle"
    * If a class string contains any priority key as a substring, map to that key.
    * Use default 0.50 ONLY if no mapping is possible.

    Scene Description JSON (verbatim string):
    {strip_md_fences(desc_text)}

    Procedure
    A) Environment scores:
    - Set Ew, Et, Er by exact lookup from the tables.
    - Compute El via lane_rules (substring match; first match wins; else 'default').
    - Round Ew, Et, Er, El to 6 decimals.

    B) Difficulty (STRICT; compute from an explicit picked block):
    - First, output a "picked" field echoing the exact numeric inputs you will use:
        "picked": {{"ww": {cfg_for_prompt["weights"]["ww"]}, "wt": {cfg_for_prompt["weights"]["wt"]},
                    "wr": {cfg_for_prompt["weights"]["wr"]}, "wl": {cfg_for_prompt["weights"]["wl"]},
                    "Ew": <Ew>, "Et": <Et>, "Er": <Er>, "El": <El>}}
    - Then compute term-by-term using ONLY those picked values:
        ww_times_Ew = picked.ww * picked.Ew
        wt_times_Et = picked.wt * picked.Et
        wr_times_Er = picked.wr * picked.Er
        wl_times_El = picked.wl * picked.El
        (Round each product to 6 decimals.)
    - Compute D = ww_times_Ew + wt_times_Et + wr_times_Er + wl_times_El (round D to 6 decimals).
    - D MUST equal the sum of those four products exactly. If not, recompute until equal.

    C) Complexity (ORDER MATTERS — build the list FIRST, then compute from it):
    - Extract critical_objects.
    - Normalize each object's class (rules above), then p = priority[class] (or 0.50 if truly unknown).
    - Build per_object = [{{"class":"<normalized>", "p":<float>}}, ...] for ALL critical objects (0 or more).
    - Set N = len(per_object).
    - Set sum_p = sum of all per_object[i].p values (exact arithmetic).
    - Compute alpha_times_N = alpha * N and beta_times_sum_p = beta * sum_p.
    - Set C = alpha_times_N + beta_times_sum_p.
    - Round sum_p, alpha_times_N, beta_times_sum_p, and C to 6 decimals.

    Self-checks (MANDATORY; numbers must satisfy equations exactly):
    - Check N == len(per_object).
    - Check sum_p == sum(per_object[*].p).
    - Check D == ww_times_Ew + wt_times_Et + wr_times_Er + wl_times_El using the values in "picked".
    - Check C == alpha*N + beta*sum_p.
    - If ANY mismatch occurs, recompute and FIX the numbers so equalities hold EXACTLY before output.

    Return ONLY this JSON:
    {{
    "difficulty": {{
        "picked": {{
        "ww": {cfg_for_prompt["weights"]["ww"]}, "wt": {cfg_for_prompt["weights"]["wt"]},
        "wr": {cfg_for_prompt["weights"]["wr"]}, "wl": {cfg_for_prompt["weights"]["wl"]},
        "Ew": <float>, "Et": <float>, "Er": <float>, "El": <float>
        }},
        "weights": {json.dumps(cfg_for_prompt["weights"])},
        "terms": {{
        "ww_times_Ew": <float>, "wt_times_Et": <float>, "wr_times_Er": <float>, "wl_times_El": <float>
        }},
        "D": <float>,
        "calculation": "D = ww*Ew + wt*Et + wr*Er + wl*El = <terms.ww_times_Ew> + <terms.wt_times_Et> + <terms.wr_times_Er> + <terms.wl_times_El> = <D>",
        "notes": "<short note on environment effects>"
    }},
    "complexity": {{
        "per_object": [{{"class":"<str>","p":<float>}}],   // build this FIRST
        "N": <int>,                                        // N = len(per_object)
        "sum_p": <float>,                                  // sum_p = sum of per_object[*].p
        "alpha": {cfg_for_prompt["alpha"]}, "beta": {cfg_for_prompt["beta"]},
        "terms": {{
        "alpha_times_N": <float>, "beta_times_sum_p": <float>
        }},
        "C": <float>,
        "calculation": "C = alpha*N + beta*sum_p = {cfg_for_prompt["alpha"]}*<N> + {cfg_for_prompt["beta"]}*<sum_p> = <terms.alpha_times_N> + <terms.beta_times_sum_p> = <C>"
    }},
    "reasoning": "<2–3 sentences on how the critical objects influence near-term driving>"
    }}
    """.strip()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model = ModelHandler(args.model, cfg)
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

    decision_json["audit"]["weather_key"] = wk
    decision_json["audit"]["scene_key"]   = sk

    llm_choice     = decision_json["policy"]["Final model"]
    shortlist_rows = _audit.get("shortlist_detail", [])
    R_llm        = float(_audit["R"])
    C_min_llm    = float(_audit["C_min"])
    C_max_llm    = float(_audit["C_max"])
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

    print("\n[DECISION] Step 3's own choice (used unconditionally -- no RL layer in this script):")
    print(f"  Final model : {llm_choice}")

    candidate_models_llm = [row.get("model") for row in shortlist_rows]
    fallback_order = [llm_choice] + [m for m in candidate_models_llm if m != llm_choice]
    exec_result = DR.run_with_fallback(fallback_order, args.image)

    if exec_result is not None:
        if exec_result.model != llm_choice:
            print(f"[EXECUTION] '{llm_choice}' unavailable; executed "
                  f"'{exec_result.model}' instead.")
        executed_model = exec_result.model
        observed_confidence = exec_result.confidence
        observed_latency_ms = exec_result.latency_ms
        decision_json["policy"]["Final model"] = executed_model

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
        learned_confidence = None
        annotated_path = None
        print("[EXECUTION] No candidate detector could be executed.")

    extra_note = (
        f" Step 3 selects {llm_choice} for this {wk}/{sk} scene "
        f"(D≈{D_const:.3f}, C≈{C_const:.3f})."
    )
    if executed_model is not None:
        extra_note += (
            f" Executed {executed_model}: confidence={observed_confidence:.3f}, "
            f"latency={observed_latency_ms:.1f}ms."
        )
    if "reasoning" in decision_json and isinstance(decision_json["reasoning"], str):
        decision_json["reasoning"] += extra_note
    else:
        decision_json["reasoning"] = extra_note

    decision_json["policy"]["_execution"] = {
        "model": executed_model,
        "confidence": observed_confidence,
        "latency_ms": observed_latency_ms,
        "num_detections": exec_result.num_detections if exec_result else None,
        "backend": exec_result.backend if exec_result else None,
        "device": exec_result.device if exec_result else None,
        "learned_confidence_profile": learned_confidence,
        "annotated_frame": annotated_path,
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
        "active_model": llm_choice,
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

    WDB.save_confidence_overrides()

    print(f"Saved CoT results -> {args.out}\n")


if __name__ == "__main__":
    main()