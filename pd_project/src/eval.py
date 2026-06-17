# src/eval.py
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import IMUWindowsDataset, build_patient_split, load_clinic_table
from .model import CNNBiLSTM
from .utils import compute_metrics


def evaluate_model(
    data_root: str,
    clinic_filename: str,
    checkpoint_path: str,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    batch_size: int = 32,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    clinic_df = load_clinic_table(os.path.join(data_root, clinic_filename))
    split_ids: Dict[str, list] = build_patient_split(clinic_df)

    test_dataset = IMUWindowsDataset(
        data_root=data_root,
        clinic_filename=clinic_filename,
        split_ids=split_ids,
        split_name="test",
        window_sec=window_sec,
        stride_sec=stride_sec,
        augment=False,
    )

    # FIX: collate_fn для (x, y, sid)
    def collate_fn(batch):
        xs, ys, sids = zip(*batch)
        return torch.stack(xs), torch.stack(ys), list(sids)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    in_channels = checkpoint["in_channels"]
    model = CNNBiLSTM(in_channels=in_channels, num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_true = []
    all_proba = []
    all_sids = []

    with torch.no_grad():
        for x_batch, y_batch, sids in tqdm(test_loader, desc="Test"):
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            proba = torch.softmax(logits, dim=1)[:, 1].cpu()

            all_true.extend(y_batch.numpy().tolist())
            all_proba.extend(proba.numpy().tolist())
            all_sids.extend(sids)

    # FIX: передаём subject_ids для patient-level метрик
    metrics = compute_metrics(all_true, all_proba, subject_ids=all_sids)

    print("Test metrics:")
    print(f"  Window Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Window F1:       {metrics['f1']:.4f}")
    print(f"  Window ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"  Best threshold:  {metrics['best_threshold']:.2f}")
    print("  Window confusion matrix:\n", metrics["confusion_matrix"])

    if "patient_f1" in metrics:
        print(f"  Patient F1:      {metrics['patient_f1']:.4f}")
        print(f"  Patient ROC-AUC: {metrics['patient_auc']:.4f}")
        print("  Patient confusion matrix:\n", metrics["patient_confusion_matrix"])
