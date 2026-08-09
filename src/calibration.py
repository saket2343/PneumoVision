"""
calibration.py
---------------
Temperature scaling: a standard, minimal post-hoc probability calibration
method (Guo et al., 2017). Fits a single scalar T > 0 on VALIDATION
predictions only, then rescales future probabilities via:

    calibrated_p = sigmoid(logit(p) / T)

T > 1 softens (de-sharpens) overconfident probabilities; T < 1 sharpens
them; T = 1 is a no-op.

This project's model outputs a sigmoid probability directly (no exposed
pre-activation logits), so logits are recovered mathematically from the
probability itself (`logit(p) = ln(p / (1-p))`), which is exact wherever
the model's own sigmoid was computed in float64-equivalent precision and a
very close approximation in practice — this is the standard workaround used
when only a probability API is available, not a source of additional error
beyond floating-point clipping at the extremes (handled via `eps` below).

IMPORTANT: fit_temperature must only ever be called with VALIDATION
(y_true, y_prob) — never test labels. Calling it with test data would mean
the "final unbiased test evaluation" is no longer unbiased.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _logits_from_probs(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _binary_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def apply_temperature(y_prob: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities by a fitted temperature. temperature=1.0 is a no-op."""
    if temperature == 1.0:
        return np.asarray(y_prob).copy()
    logits = _logits_from_probs(np.asarray(y_prob))
    return _sigmoid(logits / temperature)


def fit_temperature(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 5.0,
    t_step: float = 0.01,
) -> Tuple[float, Dict]:
    """Grid-search the temperature T in [t_min, t_max] that minimizes binary
    NLL on the given (validation!) predictions.

    Returns (best_temperature, diagnostics) where diagnostics includes the
    NLL and Brier score before/after calibration, for reporting whether
    calibration actually helped (per the audit brief's requirement to
    "explain whether calibration actually improves decision quality").
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    temperatures = np.arange(t_min, t_max + t_step / 2, t_step)
    best_t, best_nll = 1.0, _binary_nll(y_true, y_prob)

    for t in temperatures:
        calibrated = apply_temperature(y_prob, t)
        nll = _binary_nll(y_true, calibrated)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    calibrated_best = apply_temperature(y_prob, best_t)
    diagnostics = {
        "temperature": best_t,
        "nll_before": _binary_nll(y_true, y_prob),
        "nll_after": best_nll,
        "brier_before": float(np.mean((y_prob - y_true) ** 2)),
        "brier_after": float(np.mean((calibrated_best - y_true) ** 2)),
    }
    return best_t, diagnostics


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """Standard reliability-diagram data: for each probability bin, the
    fraction of positives observed vs. the mean predicted probability —
    the gap between them is the calibration error in that bin.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            bins.append({"bin_low": float(lo), "bin_high": float(hi), "count": 0, "mean_predicted": None, "observed_positive_rate": None})
            continue
        bins.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_positive_rate": float(y_true[mask].mean()),
            }
        )
    return {"bins": bins, "n_bins": n_bins}


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, save_path: Path = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = reliability_bins(y_true, y_prob, n_bins)
    xs = [b["mean_predicted"] for b in data["bins"] if b["mean_predicted"] is not None]
    ys = [b["observed_positive_rate"] for b in data["bins"] if b["mean_predicted"] is not None]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(xs, ys, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
