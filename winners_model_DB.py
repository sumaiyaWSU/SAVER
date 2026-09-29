import json
import os

from typing import Dict, List, Optional, Tuple


DATA: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
    "DINO": {
        "Sunny":   {"Highway": {"latency": 139.7, "mAP": 0.731}, "Downtown": {"latency": 142.5, "mAP": 0.712}},
        "Snow":    {"Highway": {"latency": 145.7, "mAP": 0.651}, "Downtown": {"latency": 155.6, "mAP": 0.611}},
        "Rainy":   {"Highway": {"latency": 151.2, "mAP": 0.631}, "Downtown": {"latency": 157.5, "mAP": 0.602}},
        "Foggy":   {"Highway": {"latency": 142.9, "mAP": 0.664}, "Downtown": {"latency": 148.7, "mAP": 0.643}},
        "Sand":    {"Highway": {"latency": 148.3, "mAP": 0.667}, "Downtown": {"latency": 156.1, "mAP": 0.638}},

    },

    "RT-DETR-L": {
        "Sunny":   {"Highway": {"latency": 50.1, "mAP": 0.752}, "Downtown": {"latency": 51.5, "mAP": 0.729}},
        "Snow":    {"Highway": {"latency": 57.2, "mAP": 0.691}, "Downtown": {"latency": 59.8, "mAP": 0.661}},
        "Rainy":   {"Highway": {"latency": 52.1, "mAP": 0.682}, "Downtown": {"latency": 55.9, "mAP": 0.660}},
        "Foggy":   {"Highway": {"latency": 61.2, "mAP": 0.703}, "Downtown": {"latency": 62.9, "mAP": 0.674}},
        "Sand":    {"Highway": {"latency": 56.3, "mAP": 0.682}, "Downtown": {"latency": 57.1, "mAP": 0.661}},
    },

    "Sparse-RCNN": {
        "Sunny":   {"Highway": {"latency": 107.63, "mAP": 0.768}, "Downtown": {"latency": 111.85, "mAP": 0.744}},
        "Snow":    {"Highway": {"latency": 108.98, "mAP": 0.701}, "Downtown": {"latency": 111.29, "mAP": 0.656}},
        "Rainy":   {"Highway": {"latency": 105.73, "mAP": 0.684}, "Downtown": {"latency": 109.98, "mAP": 0.620}},
        "Foggy":   {"Highway": {"latency": 109.73, "mAP": 0.720}, "Downtown": {"latency": 117.83, "mAP": 0.686}},
        "Sand":    {"Highway": {"latency": 107.46, "mAP": 0.722}, "Downtown": {"latency": 109.95, "mAP": 0.686}},
    },
        "DETR": {
        "Sunny":   {"Highway": {"latency": 128.54, "mAP": 0.780}, "Downtown": {"latency": 134.40, "mAP": 0.740}},
        "Snow":    {"Highway": {"latency": 129.13, "mAP": 0.673}, "Downtown": {"latency": 132.67, "mAP": 0.601}},
        "Rainy":   {"Highway": {"latency": 121.09, "mAP": 0.667}, "Downtown": {"latency": 127.87, "mAP": 0.590}},
        "Foggy":   {"Highway": {"latency": 120.01, "mAP": 0.748}, "Downtown": {"latency": 135.53, "mAP": 0.710}},
        "Sand":    {"Highway": {"latency": 132.98, "mAP": 0.680}, "Downtown": {"latency": 138.45, "mAP": 0.610}},
    },
        "Faster-RCNN": {
        "Sunny":   {"Highway": {"latency": 97.3, "mAP": 0.735}, "Downtown": {"latency": 101.8, "mAP": 0.719}},
        "Snow":    {"Highway": {"latency": 103.5, "mAP": 0.652}, "Downtown": {"latency": 107.1, "mAP": 0.622}},
        "Rainy":   {"Highway": {"latency": 96.8, "mAP": 0.649}, "Downtown": {"latency": 101.4, "mAP": 0.618}},
        "Foggy":   {"Highway": {"latency": 98.7, "mAP": 0.675}, "Downtown": {"latency": 106.3, "mAP": 0.643}},
        "Sand":    {"Highway": {"latency": 105.4, "mAP": 0.674}, "Downtown": {"latency": 108.2, "mAP": 0.652}},

    },
        "Mask-RCNN": {
        "Sunny":   {"Highway": {"latency": 281.28, "mAP": 0.82},  "Downtown": {"latency": 298.78, "mAP": 0.778}},
        "Snow":    {"Highway": {"latency": 301.17, "mAP": 0.75},  "Downtown": {"latency": 314.38, "mAP": 0.695}},
        "Rainy":   {"Highway": {"latency": 281.53, "mAP": 0.72},  "Downtown": {"latency": 289.17, "mAP": 0.68}},
        "Foggy":   {"Highway": {"latency": 301.02, "mAP": 0.76},  "Downtown": {"latency": 310.35, "mAP": 0.712}},
        "Sand":    {"Highway": {"latency": 291.22, "mAP": 0.78},  "Downtown": {"latency": 312.33, "mAP": 0.71}},
    },
        "YOLOv11(Large)": {
        "Sunny":   {"Highway": {"latency": 35.8, "mAP": 0.727}, "Downtown": {"latency": 36.9, "mAP": 0.709}},
        "Snow":    {"Highway": {"latency": 29.3, "mAP": 0.672}, "Downtown": {"latency": 31.6, "mAP": 0.621}},
        "Rainy":   {"Highway": {"latency": 35.7, "mAP": 0.663}, "Downtown": {"latency": 38.3, "mAP": 0.614}},
        "Foggy":   {"Highway": {"latency": 31.4, "mAP": 0.680}, "Downtown": {"latency": 32.8, "mAP": 0.630}},
        "Sand":    {"Highway": {"latency": 36.5, "mAP": 0.673}, "Downtown": {"latency": 38.9, "mAP": 0.659}},
    },
        "Deformable DETR (R50)": {
        "Sunny":   {"Highway": {"latency": 296.70, "mAP": 0.732}, "Downtown": {"latency": 302.40, "mAP": 0.712}},
        "Snow":    {"Highway": {"latency": 286.30, "mAP": 0.707}, "Downtown": {"latency": 292.90, "mAP": 0.695}},
        "Rainy":   {"Highway": {"latency": 303.70, "mAP": 0.681}, "Downtown": {"latency": 310.90, "mAP": 0.651}},
        "Foggy":   {"Highway": {"latency": 289.10, "mAP": 0.697}, "Downtown": {"latency": 291.60, "mAP": 0.672}},
        "Sand":    {"Highway": {"latency": 283.40, "mAP": 0.713}, "Downtown": {"latency": 288.50, "mAP": 0.686}},
    },


}


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIDENCE_OVERRIDES_PATH = os.environ.get(
    "SAVER_CONFIDENCE_OVERRIDES",
    os.path.join(_MODULE_DIR, "winner_model_confidence.json"),
)

CONFIDENCE_DEFAULTS_PATH = os.path.join(_MODULE_DIR, "winner_model_confidence.example.json")


def _load_confidence_overrides(path: str) -> Dict:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k != "_comment"}
    return {}


_CONFIDENCE_OVERRIDES = _load_confidence_overrides(CONFIDENCE_OVERRIDES_PATH)
_CONFIDENCE_FROM_DEFAULTS = False

if not _CONFIDENCE_OVERRIDES:
    _CONFIDENCE_OVERRIDES = _load_confidence_overrides(CONFIDENCE_DEFAULTS_PATH)
    _CONFIDENCE_FROM_DEFAULTS = bool(_CONFIDENCE_OVERRIDES)
    if _CONFIDENCE_FROM_DEFAULTS:
        print(
            f"[winners_model_DB] No local override at '{CONFIDENCE_OVERRIDES_PATH}' -- "
            f"using the shipped default confidence profile from "
            f"'{CONFIDENCE_DEFAULTS_PATH}'. Any real observation recorded via "
            "record_confidence_observation() + save_confidence_overrides() is "
            "written to the local override path instead, so it takes "
            "precedence over these defaults from then on without editing the "
            "repo's shipped file."
        )

if not _CONFIDENCE_OVERRIDES:
    print(
        "[winners_model_DB] NOTE: no confidence data available for any "
        f"(model, weather, scene) cell (looked in '{CONFIDENCE_OVERRIDES_PATH}' "
        f"and '{CONFIDENCE_DEFAULTS_PATH}'). Candidate shortlisting will use "
        "latency + mAP only (matching the paper's Table VI profiles) until "
        "real confidence data exists -- either supplied by hand, or learned "
        "automatically as RL_integrated-CoT.py actually executes detectors "
        "and calls record_confidence_observation(). Each cell starts using "
        "confidence as soon as it has at least one real observation; cells "
        "with none keep falling back to latency + mAP. This does NOT affect "
        "the RL reward's confidence gate, which always uses the real "
        "confidence measured from actually running the detector."
    )


def get_confidence(model: str, weather: str, scene: str) -> Optional[float]:
    """Real, profiled confidence for (model, weather, scene) -- either one
    you supplied by hand via winner_model_confidence.json, or one learned
    online from actual detector_runner.py executions via
    record_confidence_observation(). Returns None otherwise -- never
    fabricates a value from mAP or anything else in DATA."""
    entry = (_CONFIDENCE_OVERRIDES.get(model, {}) or {}).get(weather, {})
    entry = (entry or {}).get(scene)
    return float(entry) if entry is not None else None


CONFIDENCE_LEARNING_RATE = 0.3


def record_confidence_observation(
    model: str, weather: str, scene: str, confidence: float,
    alpha: float = CONFIDENCE_LEARNING_RATE,
) -> float:
    """Folds ONE real, measured confidence value (from actually running
    `model` via detector_runner.py on this weather/scene) into the learned
    profile for that cell. First observation seeds the value outright;
    later ones blend in via EMA so the estimate tracks recent/"latest"
    performance. This is the ONLY way a confidence value enters this
    module -- it is always something detector_runner.py actually measured,
    never derived from mAP or guessed. Call save_confidence_overrides()
    to persist the result to disk (see winner_model_confidence.json).
    """
    per_model = _CONFIDENCE_OVERRIDES.setdefault(model, {})
    per_weather = per_model.setdefault(weather, {})
    old = per_weather.get(scene)
    new_value = float(confidence) if old is None else old + alpha * (float(confidence) - old)
    per_weather[scene] = new_value
    return new_value


def save_confidence_overrides(path: Optional[str] = None) -> None:
    """Persists the learned confidence profile to disk (default:
    winner_model_confidence.json / SAVER_CONFIDENCE_OVERRIDES) so it
    accumulates across runs, the same way rl_memory.json persists RL_Q."""
    target = path or CONFIDENCE_OVERRIDES_PATH
    to_save = {k: v for k, v in _CONFIDENCE_OVERRIDES.items() if k != "_comment"}
    with open(target, "w") as f:
        json.dump(to_save, f, indent=2)


WEATHER_MAP = {
    "snow": "Snow",
    "rain": "Rainy", "rainy": "Rainy", "rainstorm": "Rainy",
    "fog": "Foggy", "foggy": "Foggy",
    "sand": "Sand", "dust": "Sand", "sandstorm": "Sand",
    "sunny": "Sunny", "clear": "Sunny", "overcast": "Sunny", "cloudy": "Sunny",
}

def map_weather(w: str) -> str:
    w = (w or "").strip().lower()
    return WEATHER_MAP.get(w, "Sunny")  

def map_scene(road: str) -> str:
    s = (road or "").lower()
    if "highway" in s or "ramp" in s:
        return "Highway"
    return "Downtown"

def all_models() -> List[str]:
    return list(DATA.keys())

def metrics(model: str, weather: str, scene: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (latency, mAP, confidence). confidence is None unless you've
    supplied a real profiled value (see get_confidence())."""
    try:
        m = DATA[model][weather][scene]
        return m["latency"], m["mAP"], get_confidence(model, weather, scene)
    except Exception:
        return None, None, None

def shortlist_candidates(
    weather: str,
    scene: str,
    max_latency_ms: float = None,
    min_map: float = None,
    min_confidence: float = None,
    top_k: int = 3,
    tradeoff: float = 0.5,
) -> List[Dict]:

    rows = []
    for m in all_models():
        lat, mp, conf = metrics(m, weather, scene)
        if lat is None or mp is None:
            continue
        if (max_latency_ms is not None and lat > max_latency_ms):
            continue
        if (min_map is not None and mp < min_map):
            continue
        if (min_confidence is not None and conf is not None and conf < min_confidence):
            continue
        rows.append({"model": m, "latency": lat, "mAP": mp, "confidence": conf})

    if not rows:
        return []

    lats = [r["latency"] for r in rows]
    maps = [r["mAP"] for r in rows]
    lat_min, lat_max = min(lats), max(lats)
    map_min, map_max = min(maps), max(maps)

    def norm(v, lo, hi):
        if hi == lo: return 0.5
        return (v - lo) / (hi - lo)

    for r in rows:
        n_m = norm(r["mAP"], map_min, map_max)          
        n_l = norm(r["latency"], lat_min, lat_max)      
        r["score"] = tradeoff * n_m - (1 - tradeoff) * n_l

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_k]