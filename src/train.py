"""
train.py — Evaluation metrics for RUL prediction.
"""

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, and NASA scoring function.
    
    Parameters
    ----------
    y_true : np.ndarray
        True RUL values
    y_pred : np.ndarray
        Predicted RUL values
    
    Returns
    -------
    dict with keys: MAE, RMSE, NASA_Score
    """
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # NASA scoring function (asymmetric — penalizes late predictions more)
    scores = []
    for e in errors:
        if e < 0:
            scores.append(np.exp(-e / 13.0) - 1)
        else:
            scores.append(np.exp(e / 10.0) - 1)
    nasa_score = np.sum(scores)

    return {"MAE": mae, "RMSE": rmse, "NASA_Score": nasa_score}
