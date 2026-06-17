# src/dataset.py
import os
import glob
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# пять сенсоров
SENSOR_PREFIXES = ["ch", "lh", "ll", "rh", "rl"]  # chest, left hand, left leg, right hand, right leg


def load_clinic_table(clinic_path: str) -> pd.DataFrame:
    """
    Читает Clinic_DataPDBioStampRCStudy.csv и добавляет бинарную метку label: Control -> 0, PD -> 1.
    """
    df = pd.read_csv(clinic_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    if "ID" not in df.columns or "Status" not in df.columns:
        raise ValueError("Clinic file must have columns 'ID' and 'Status'")

    df["Status"] = df["Status"].astype(str).str.strip()
    status_map = {"Control": 0, "PD": 1}
    df["label"] = df["Status"].map(status_map)

    if df["label"].isna().any():
        bad = df[df["label"].isna()]["Status"].unique()
        raise ValueError(f"Unknown Status values in clinic file: {bad}")

    return df


def load_subject_signals(subject_dir: str, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает сигналы пациента с фиксированным числом каналов:
    5 сенсоров * 3 оси = 15 каналов.
    Если сенсор отсутствует, его 3 канала заполняются нулями.
    Все сенсоры обрезаются до общей минимальной длины среди доступных.
    """
    sensor_data = {}
    timestamps_ref = None
    min_len = None

    for prefix in SENSOR_PREFIXES:
        pattern = os.path.join(subject_dir, f"{prefix}_ID{subject_id}Accel.csv")
        matches = glob.glob(pattern)
        if len(matches) == 0:
            continue

        csv_path = matches[0]
        df = pd.read_csv(csv_path)
        df = df.rename(columns={c: c.strip() for c in df.columns})

        if df.shape[1] < 4:
            raise ValueError(f"Expected at least 4 columns in {csv_path}, got {df.shape[1]}")

        df = df.iloc[:, :4].copy()
        df.columns = ["Timestamp", "ax", "ay", "az"]

        ts = df["Timestamp"].to_numpy(dtype=np.float64)
        sig = df[["ax", "ay", "az"]].to_numpy(dtype=np.float32)

        sensor_data[prefix] = sig

        if timestamps_ref is None:
            timestamps_ref = ts
            min_len = len(df)
        else:
            min_len = min(min_len, len(df))

    if len(sensor_data) == 0:
        raise FileNotFoundError(f"No accel files found for subject {subject_id} in {subject_dir}")

    if min_len is None or min_len <= 1:
        raise ValueError(f"Too few samples for subject {subject_id}")

    channel_blocks = []
    for prefix in SENSOR_PREFIXES:
        if prefix in sensor_data:
            block = sensor_data[prefix][:min_len]  # (N, 3)
        else:
            block = np.zeros((min_len, 3), dtype=np.float32)  # отсутствующий сенсор
        channel_blocks.append(block)

    timestamps = timestamps_ref[:min_len]
    signals = np.concatenate(channel_blocks, axis=1)  # (N, 15)

    return timestamps, signals


def segment_windows(
    timestamps: np.ndarray,
    signals: np.ndarray,
    window_sec: float,
    stride_sec: float,
    fs: float,
) -> List[np.ndarray]:
    """
    Разбивает сигнал на окна длиной window_sec c шагом stride_sec.
    Возвращает список numpy-массивов формы (C, L).
    """
    C = signals.shape[1]
    L = int(window_sec * fs)
    step = int(stride_sec * fs)
    if L <= 0 or step <= 0:
        raise ValueError("window_sec and stride_sec must give positive lengths")

    windows = []
    for start in range(0, signals.shape[0] - L + 1, step):
        end = start + L
        seg = signals[start:end, :]  # (L, C)
        if seg.shape[0] == L:
            windows.append(seg.T)  # (C, L)

    return windows


def build_patient_split(
    clinic_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, List[str]]:
    """
    Делит список субъектов по patient-wise схеме: train/val/test
    со стратификацией по метке.
    Возвращает словарь с ключами 'train', 'val', 'test' и списками ID (строки без zero-padding).
    """
    assert "ID" in clinic_df.columns
    assert "label" in clinic_df.columns

    ids = clinic_df["ID"].astype(str).values
    labels = clinic_df["label"].values

    ids_trainval, ids_test, labels_trainval, _ = train_test_split(
        ids, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    val_rel_size = val_size / (1.0 - test_size)
    ids_train, ids_val, _, _ = train_test_split(
        ids_trainval,
        labels_trainval,
        test_size=val_rel_size,
        stratify=labels_trainval,
        random_state=random_state,
    )

    return {
        "train": list(ids_train),
        "val": list(ids_val),
        "test": list(ids_test),
    }


class IMUWindowsDataset(Dataset):
    """
    PyTorch Dataset, который:
      - читает клинику,
      - выбирает ID, относящиеся к нужному split'у,
      - грузит все доступные сенсоры по этому ID,
      - режет на окна (C, L),
      - сопоставляет каждой окне метку пациента.

    __getitem__ возвращает (x, y, subject_id) — subject_id нужен для patient-level метрик.
    """

    def __init__(
        self,
        data_root: str,
        clinic_filename: str,
        split_ids: Dict[str, List[str]],
        split_name: str,
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
        augment: bool = False,
        fs: float = 31.25,
    ):
        super().__init__()
        self.data_root = data_root
        self.clinic_path = os.path.join(data_root, clinic_filename)
        self.clinic_df = load_clinic_table(self.clinic_path)

        self.split_ids = split_ids
        self.split_name = split_name
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.augment = augment
        self.fs = fs

        # (window_array, label, subject_id)
        self.samples: List[Tuple[np.ndarray, int, str]] = []

        min_len = int(self.window_sec * self.fs)

        for sid in self.split_ids[self.split_name]:
            sid_str = str(sid).zfill(3)  # папка '005', '007', ...

            subj_dir = os.path.join(self.data_root, "FullDataSet_PD-BioStampRC21", sid_str)
            if not os.path.isdir(subj_dir):
                continue

            try:
                timestamps, signals = load_subject_signals(subj_dir, sid_str)
            except FileNotFoundError:
                continue

            if signals.shape[0] < min_len:
                continue

            windows = segment_windows(
                timestamps,
                signals,
                self.window_sec,
                self.stride_sec,
                fs=self.fs,
            )

            label_row = self.clinic_df[self.clinic_df["ID"].astype(str) == str(sid)]
            if len(label_row) == 0:
                continue
            label = int(label_row["label"].iloc[0])

            for w in windows:
                self.samples.append((w, label, str(sid)))

        if len(self.samples) == 0:
            raise ValueError(
                f"No windows were generated for split '{self.split_name}'. "
                f"Check paths and window_sec/stride_sec."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, sid = self.samples[idx]  # x: (C, L) numpy, y: int, sid: str
        x = torch.from_numpy(x).float()

        # FIX: z-score нормализация по каждому каналу
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-8)
        x = (x - mean) / std

        y = torch.tensor(y, dtype=torch.long)

        if self.augment and self.split_name == "train":
            x = self._augment(x)

        return x, y, sid

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise
        noise = torch.randn_like(x) * 0.01
        # Random scaling (лёгкое масштабирование)
        scale = torch.empty(x.shape[0], 1).uniform_(0.95, 1.05)
        return x * scale + noise
