import os
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import CNNBiLSTM
from .utils import compute_metrics  # единственный источник — utils.py
from .dataset import IMUWindowsDataset, build_patient_split, load_clinic_table


def train_model(
    data_root: str,
    clinic_filename: str,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    batch_size: int = 32,
    lr: float = 3e-4,
    num_epochs: int = 50,
    device: str = None,
    checkpoint_dir: str = "checkpoints",
    max_train_patients: int | None = None,
    max_val_patients: int | None = None,
    val_size: float = 0.20,
    test_size: float = 0.20,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(checkpoint_dir, exist_ok=True)

    clinic_df = load_clinic_table(os.path.join(data_root, clinic_filename))
    split_ids: Dict[str, list] = build_patient_split(clinic_df, val_size=val_size, test_size=test_size)

    if max_train_patients is not None:
        split_ids["train"] = split_ids["train"][:max_train_patients]
    if max_val_patients is not None:
        split_ids["val"] = split_ids["val"][:max_val_patients]

    print(f"Train patients: {len(split_ids['train'])}")
    print(f"Val patients: {len(split_ids['val'])}")
    print("Train IDs:", split_ids["train"])
    print("Val IDs:", split_ids["val"])

    train_dataset = IMUWindowsDataset(
        data_root=data_root,
        clinic_filename=clinic_filename,
        split_ids=split_ids,
        split_name="train",
        window_sec=window_sec,
        stride_sec=stride_sec,
        augment=True,
    )

    val_dataset = IMUWindowsDataset(
        data_root=data_root,
        clinic_filename=clinic_filename,
        split_ids=split_ids,
        split_name="val",
        window_sec=window_sec,
        stride_sec=stride_sec,
        augment=False,
    )

    print(f"Train windows: {len(train_dataset)}")
    print(f"Val windows: {len(val_dataset)}")

    # FIX: dataset теперь возвращает (x, y, sid) — collate_fn распаковывает sid отдельно
    def collate_fn(batch):
        xs, ys, sids = zip(*batch)
        return torch.stack(xs), torch.stack(ys), list(sids)

    sample_x, _, _ = train_dataset[0]
    in_channels = sample_x.shape[0]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = CNNBiLSTM(in_channels=in_channels, num_classes=2).to(device)

    train_labels = [lbl for _, lbl, _ in train_dataset.samples]
    class_counts = torch.bincount(torch.tensor(train_labels), minlength=2)
    class_weights = 1.0 / class_counts.float().clamp_min(1.0)
    class_weights = class_weights / class_weights.sum() * 2.0
    print("Class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = Adam(model.parameters(), lr=lr)

    best_metric = -1.0
    best_path = os.path.join(checkpoint_dir, "best_model.pt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        # FIX: распаковываем три значения из батча
        for x_batch, y_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= len(train_dataset)

        model.eval()
        all_true = []
        all_proba = []
        all_sids = []  # FIX: собираем subject_ids для patient-level метрик

        with torch.no_grad():
            for x_batch, y_batch, sids in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x_batch = x_batch.to(device, non_blocking=True)
                logits = model(x_batch)
                proba = torch.softmax(logits, dim=1)[:, 1].cpu()

                all_true.extend(y_batch.numpy().tolist())
                all_proba.extend(proba.numpy().tolist())
                all_sids.extend(sids)  # FIX

        # FIX: передаём subject_ids — compute_metrics сам посчитает patient-level
        metrics = compute_metrics(all_true, all_proba, subject_ids=all_sids)

        print(
            f"Epoch {epoch}: train_loss={epoch_loss:.4f}, "
            f"window_f1={metrics.get('f1', 0.0):.4f}, "
            f"window_auc={metrics.get('roc_auc', 0.0):.4f}, "
            f"patient_f1={metrics.get('patient_f1', 0.0):.4f}, "
            f"patient_auc={metrics.get('patient_auc', 0.0):.4f}, "
            f"thr={metrics.get('best_threshold', 0.5):.2f}"
        )
        print("Window confusion matrix:\n", metrics.get("confusion_matrix"))
        if "patient_confusion_matrix" in metrics:
            print("Patient confusion matrix:\n", metrics["patient_confusion_matrix"])

        # FIX: сохраняем по patient_f1 если есть, иначе по window f1
        current_metric = metrics.get("patient_f1", metrics.get("f1", 0.0))
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "metrics": metrics,
                    "split_ids": split_ids,
                    "window_sec": window_sec,
                    "stride_sec": stride_sec,
                },
                best_path,
            )
            print(f"New best model saved to {best_path}")

    print("Training finished. Best monitored metric:", best_metric)
