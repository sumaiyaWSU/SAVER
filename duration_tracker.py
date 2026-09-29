import json
import math
import os
from typing import Any, Dict, List, Optional

DURATION_STATE_PATH = os.environ.get(
    "SAVER_DURATION_STATE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "duration_state.json"),
)

DEFAULT_TAU_MIN = 1
DEFAULT_TAU_MAX = 10

WEATHER_VOCAB = ["Sunny", "Rainy", "Snow", "Foggy", "Sand"]
SCENE_VOCAB = ["Highway", "Downtown"]
TIME_VOCAB = ["day", "dusk", "night", "dawn"]


def _one_hot(value: str, vocab: List[str]) -> List[float]:
    return [1.0 if value == v else 0.0 for v in vocab]


def build_feature_vector(
    weather_key: str, scene_key: str, time_of_day: str, D: float, C: float, C_max: float
) -> List[float]:
    """F_t = phi(z_t): one-hot categorical fields (weather_key, scene_key,
    time_of_day) concatenated with normalized numerical fields (D, C).
    D is already ~[0,1] by construction (Sec. III.B.2 Difficulty Score
    Estimation formula);
    C is normalized by the running C_max the same way C_hat_t is below, so
    the feature vector and the duration formula stay consistent with each
    other."""
    c_norm = 0.0 if C_max <= 0 else max(0.0, min(1.0, C / C_max))
    d_norm = max(0.0, min(1.0, D))
    return (
        _one_hot(weather_key, WEATHER_VOCAB)
        + _one_hot(scene_key, SCENE_VOCAB)
        + _one_hot((time_of_day or "").strip().lower(), TIME_VOCAB)
        + [d_norm, c_norm]
    )


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """S_hat_t. Returns 0.0 (treated as "no similarity established") if
    either vector is missing/empty/zero rather than raising -- this is the
    correct behavior for the very first frame of a sequence, which has no
    F_{t-1} to compare against."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def compute_duration(
    similarity: float, c_hat: float,
    tau_min: int = DEFAULT_TAU_MIN, tau_max: int = DEFAULT_TAU_MAX,
) -> int:
    """tau_t = tau_min + (tau_max - tau_min) * S_hat_t * (1 - C_hat_t),
    rounded to the nearest whole frame. Bounded to [tau_min, tau_max] by
    construction since similarity/c_hat are each clamped to [0, 1]."""
    similarity = max(0.0, min(1.0, similarity))
    c_hat = max(0.0, min(1.0, c_hat))
    tau = tau_min + (tau_max - tau_min) * similarity * (1.0 - c_hat)
    return max(tau_min, round(tau))


def load_state(path: str = DURATION_STATE_PATH) -> Optional[Dict[str, Any]]:
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_state(state: Dict[str, Any], path: str = DURATION_STATE_PATH) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def clear_state(path: str = DURATION_STATE_PATH) -> None:
    """Starts a fresh sequence: no prior frame to compare against, no
    active model to reuse, running C_max resets. Call this for the first
    frame of a new, unrelated sequence (see --new-sequence)."""
    if path and os.path.exists(path):
        os.remove(path)