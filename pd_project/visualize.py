# visualize.py
"""
Строит графики:
  1. ROC-кривая (window-level)
  2. Precision-Recall кривая (window-level)
  3. Confusion matrix — window-level
  4. Confusion matrix — patient-level
  5. Patient bar chart: вероятность PD по каждому пациенту

Использование:
  python visualize.py \
    --data_root "/Users/neverlast/education/vuz/kp ml/pd_project" \
    --clinic_file Clinic_DataPDBioStampRCStudy.csv \
    --checkpoint checkpoints/best_model.pt \
    --out_dir figures
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)

from src.dataset import IMUWindowsDataset, build_patient_split, load_clinic_table
from src.model import CNNBiLSTM
from src.utils import aggregate_patient_predictions, find_best_threshold


# ── стиль ──────────────────────────────────────────────────────────────────
COLORS = {"pd": "#E05C5C", "ctrl": "#5C8FE0", "neutral": "#555555"}
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def collate_fn(batch):
    xs, ys, sids = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(sids)


def run_inference(model, loader, device):
    model.eval()
    all_true, all_proba, all_sids = [], [], []
    with torch.no_grad():
        for x_batch, y_batch, sids in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_true.extend(y_batch.numpy().tolist())
            all_proba.extend(proba.tolist())
            all_sids.extend(sids)
    return np.array(all_true), np.array(all_proba), all_sids


# ── 1. ROC-кривая ──────────────────────────────────────────────────────────
def plot_roc(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=COLORS["pd"], lw=2,
            label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["neutral"], lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (window-level)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}  (AUC={roc_auc:.3f})")


# ── 2. Precision-Recall кривая ─────────────────────────────────────────────
def plot_pr(y_true, y_proba, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, color=COLORS["ctrl"], lw=2,
            label=f"AP = {ap:.3f}")
    baseline = y_true.mean()
    ax.axhline(baseline, linestyle="--", color=COLORS["neutral"], lw=1,
               label=f"Baseline = {baseline:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (window-level)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}  (AP={ap:.3f})")


# ── 3 & 4. Confusion matrix ────────────────────────────────────────────────
def plot_cm(cm, title, class_names, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── 5. Patient bar chart ───────────────────────────────────────────────────
def plot_patient_bars(patient_ids, patient_true, patient_proba, best_thr, out_path):
    order = np.argsort(patient_proba)
    ids_s = [patient_ids[i] for i in order]
    true_s = [patient_true[i] for i in order]
    proba_s = [patient_proba[i] for i in order]

    colors = [COLORS["pd"] if t == 1 else COLORS["ctrl"] for t in true_s]

    fig, ax = plt.subplots(figsize=(max(6, len(ids_s) * 0.7), 4))
    bars = ax.bar(range(len(ids_s)), proba_s, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(best_thr, linestyle="--", color=COLORS["neutral"], lw=1.5,
               label=f"Threshold = {best_thr:.2f}")

    ax.set_xticks(range(len(ids_s)))
    ax.set_xticklabels([f"#{i}" for i in ids_s], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("P(PD)")
    ax.set_ylim(0, 1)
    ax.set_title("Patient-level PD probability")

    # легенда вручную
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["pd"], label="PD (true)"),
        Patch(facecolor=COLORS["ctrl"], label="Control (true)"),
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], linestyle="--", color=COLORS["neutral"],
                   label=f"Threshold={best_thr:.2f}")
    ], loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--clinic_file", default="Clinic_DataPDBioStampRCStudy.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Какой сплит использовать для графиков (default: val)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- загружаем модель ---
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    in_channels = checkpoint["in_channels"]
    window_sec = checkpoint.get("window_sec", 10.0)
    stride_sec = checkpoint.get("stride_sec", 5.0)

    model = CNNBiLSTM(in_channels=in_channels, num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- датасет ---
    clinic_df = load_clinic_table(os.path.join(args.data_root, args.clinic_file))
    split_ids = build_patient_split(
        clinic_df, val_size=args.val_size, test_size=args.test_size
    )

    dataset = IMUWindowsDataset(
        data_root=args.data_root,
        clinic_filename=args.clinic_file,
        split_ids=split_ids,
        split_name=args.split,
        window_sec=window_sec,
        stride_sec=stride_sec,
        augment=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn,
    )
    print(f"Split: {args.split} | Windows: {len(dataset)}")

    # --- инференс ---
    y_true, y_proba, sids = run_inference(model, loader, device)
    print(f"Inference done. Windows: {len(y_true)}")

    # --- patient-level агрегация ---
    patient_ids, patient_true, patient_proba = aggregate_patient_predictions(
        sids, y_true.tolist(), y_proba.tolist()
    )
    best_thr, _ = find_best_threshold(patient_true, patient_proba)

    # --- строим графики ---
    print("\nBuilding figures...")
    plot_roc(y_true, y_proba,
             os.path.join(args.out_dir, "roc_curve.png"))

    plot_pr(y_true, y_proba,
            os.path.join(args.out_dir, "pr_curve.png"))

    # window-level CM
    best_thr_win, _ = find_best_threshold(y_true.tolist(), y_proba.tolist())
    y_pred_win = (y_proba >= best_thr_win).astype(int)
    cm_win = confusion_matrix(y_true, y_pred_win)
    plot_cm(cm_win, "Confusion Matrix (window-level)",
            ["Control", "PD"],
            os.path.join(args.out_dir, "cm_window.png"))

    # patient-level CM
    pt_pred = (np.array(patient_proba) >= best_thr).astype(int)
    cm_pat = confusion_matrix(patient_true, pt_pred)
    plot_cm(cm_pat, "Confusion Matrix (patient-level)",
            ["Control", "PD"],
            os.path.join(args.out_dir, "cm_patient.png"))

    plot_patient_bars(patient_ids, patient_true, patient_proba, best_thr,
                      os.path.join(args.out_dir, "patient_bars.png"))

    print(f"\nAll figures saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
