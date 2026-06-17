# src/utils.py
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def find_best_threshold(y_true: List[int], y_proba: List[float]) -> Tuple[float, float]:
    """Перебирает пороги от 0.1 до 0.9 и возвращает (best_thr, best_f1)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_true, (y_proba >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def aggregate_patient_predictions(
    subject_ids: List[str],
    y_true: List[int],
    y_proba: List[float],
) -> Tuple[List[str], List[int], List[float]]:
    """
    Усредняет предсказания окон по каждому пациенту.
    Возвращает (patient_ids, patient_true, patient_proba).
    """
    patient_dict = defaultdict(lambda: {"proba": [], "label": None})

    for sid, yt, yp in zip(subject_ids, y_true, y_proba):
        patient_dict[sid]["proba"].append(float(yp))
        patient_dict[sid]["label"] = int(yt)

    patient_ids, patient_true, patient_proba = [], [], []
    for sid, v in patient_dict.items():
        patient_ids.append(sid)
        patient_true.append(v["label"])
        patient_proba.append(float(np.mean(v["proba"])))

    return patient_ids, patient_true, patient_proba


def compute_metrics(
    y_true: List[int],
    y_proba: List[float],
    threshold: float = 0.5,
    subject_ids: List[str] = None,
) -> dict:
    """
    Считает window-level метрики.
    Если передан subject_ids — дополнительно считает patient-level метрики.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Ищем лучший порог
    best_thr, _ = find_best_threshold(y_true.tolist(), y_proba.tolist())
    y_pred = (y_proba >= best_thr).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "best_threshold": best_thr,
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    # Patient-level метрики
    if subject_ids is not None:
        _, pt_true, pt_proba = aggregate_patient_predictions(
            subject_ids, y_true.tolist(), y_proba.tolist()
        )
        pt_true = np.asarray(pt_true)
        pt_proba = np.asarray(pt_proba)

        pt_best_thr, _ = find_best_threshold(pt_true.tolist(), pt_proba.tolist())
        pt_pred = (pt_proba >= pt_best_thr).astype(int)

        metrics["patient_f1"] = f1_score(pt_true, pt_pred, zero_division=0)
        metrics["patient_confusion_matrix"] = confusion_matrix(pt_true, pt_pred)
        try:
            metrics["patient_auc"] = roc_auc_score(pt_true, pt_proba)
        except ValueError:
            metrics["patient_auc"] = float("nan")

    return metrics
