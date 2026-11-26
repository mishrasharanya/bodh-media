"""Train and measure regularized logistic baselines with no pretrained weights."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from bodh.audio import MEL_BANDS

CUES = ("duration_s", "mono_rms_dbfs", "low_energy_frame_fraction")
MEL = tuple(f"log_mel_mean_{i:02}" for i in range(MEL_BANDS))
FEATURES = {"cues": CUES, "log_mel": MEL, "cues_plus_log_mel": CUES + MEL}
PENALTY = 0.01


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - center) / scale

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        weights, intercept = params[:-1], params[-1]
        logits = z @ weights + intercept
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        loss += PENALTY * (weights @ weights) / 2
        residual = expit(logits) - y
        gradient = (z.T @ residual) / len(y) + PENALTY * weights
        return loss, np.r_[gradient, residual.mean()]

    solution = minimize(objective, np.zeros(z.shape[1] + 1), jac=True, method="L-BFGS-B",
                        options={"gtol": 1e-8, "ftol": 1e-12})
    if not solution.success:
        raise ValueError(f"Logistic optimization failed: {solution.message}")
    return {
        "center": center.tolist(), "scale": scale.tolist(),
        "weights": solution.x[:-1].tolist(), "intercept": float(solution.x[-1]),
    }


def scores(model: dict, x: np.ndarray) -> np.ndarray:
    return ((x - model["center"]) / model["scale"]) @ model["weights"] + model["intercept"]


def operating_curve(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, ...]:
    if not np.isfinite(score).all() or set(y) != {0, 1}:
        raise ValueError("Metrics require finite scores and both binary classes")
    order = np.argsort(-score, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(score[order])), len(y) - 1]
    positives = np.cumsum(y[order])[ends]
    negatives = ends + 1 - positives
    fpr = np.r_[0.0, negatives / negatives[-1]]
    tpr = np.r_[0.0, positives / positives[-1]]
    thresholds = np.r_[np.nextafter(score.max(), np.inf), score[order][ends]]
    return fpr, tpr, thresholds


def threshold_on_dev(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = operating_curve(y, score)
    # Thresholds are descending, so the first minimum favors fewer false alarms.
    error = (fpr + (1 - tpr)) / 2
    index = np.flatnonzero(error <= error.min() + 1e-12)[0]
    return float(thresholds[index])


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    fpr, tpr, _ = operating_curve(y, score)
    negative, positive = y == 0, y == 1
    prediction = score >= threshold
    fp = int((prediction & negative).sum())
    fn = int((~prediction & positive).sum())
    n_bonafide, n_spoof = int(negative.sum()), int(positive.sum())
    fpr_at_threshold, fnr_at_threshold = fp / n_bonafide, fn / n_spoof
    return {
        "n_bonafide": n_bonafide, "n_spoof": n_spoof,
        "roc_auc": float(np.trapz(tpr, fpr)),
        "eer": float(np.interp(0.0, fpr - (1 - tpr), fpr)),
        "false_positives": fp, "false_negatives": fn,
        "balanced_accuracy": 1 - (fpr_at_threshold + fnr_at_threshold) / 2,
    }
