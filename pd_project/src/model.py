# src/model.py
import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, C, L)
        feats = self.feature_extractor(x)  # (B, 256, L')
        feats = feats.permute(0, 2, 1)    # (B, L', 256)
        lstm_out, _ = self.lstm(feats)    # (B, L', 256)
        # Используем mean pooling по времени
        pooled = lstm_out.mean(dim=1)     # (B, 256)
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits