"""CNN-1D + BiLSTM model for 12-lead ECG classification.

NOTE (24/07): rewritten to match the architecture actually used for the
Kaggle training run that produced models/best_model.pth. The previous
local version nested each CNN block in a `CNNBlock` submodule
(cnn.0.block.0.weight, ...) and named the classifier head `self.classifier`.
The Kaggle-trained checkpoint uses flat layers per block instead
(cnn.0.weight, cnn.1.weight [BatchNorm], ... cnn.5.weight for block 2's
conv, cnn.10.weight for block 3's conv) and names the head `self.fc`.
Loading the checkpoint requires this file's module structure to match
EXACTLY, key-for-key -- that's what this rewrite does. It does not change
what the model computes for a given input, only how that computation is
organized into named submodules.

Pooling strategy: mean over the full BiLSTM output sequence (both
directions), not a concat of final hidden states. This is what the
Kaggle run was actually trained with, so it's kept as-is to match the
checkpoint -- a legitimate alternative design, not a bug to "fix" here.

IMPORTANT: whatever notebook/script you use for the NEXT Kaggle training
run (with the corrected GSVT class) must use this exact same class
definition, or you'll hit this identical mismatch again after retraining.
Keep this file in sync with whatever model definition runs on Kaggle.
"""

import torch
import torch.nn as nn


class CNN_BiLSTM(nn.Module):
    def __init__(self, num_leads=12, cnn_channels=[64, 128, 256], cnn_kernel_sizes=[7, 5, 3],
                 lstm_hidden_size=128, lstm_num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()

        # Flat CNN feature extractor -- each block's layers appended
        # directly into one nn.Sequential, not wrapped in a per-block
        # submodule. This layout is what the checkpoint's keys expect:
        # cnn.0-4 = block 1 (conv, bn, relu, pool, dropout)
        # cnn.5-9 = block 2
        # cnn.10-14 = block 3
        layers = []
        c_in = num_leads
        for c_out, k in zip(cnn_channels, cnn_kernel_sizes):
            layers.extend([
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            c_in = c_out
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        # Named `fc` (not `classifier`) to match the checkpoint's keys.
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, 12, 5000)
        features = self.cnn(x)
        features = features.permute(0, 2, 1)  # (batch, seq, channels)

        lstm_out, _ = self.lstm(features)  # (batch, seq, 2*hidden)
        pooled = torch.mean(lstm_out, dim=1)  # mean-pool over time, both directions

        return self.fc(pooled)


def build_model(config: dict) -> CNN_BiLSTM:
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})

    return CNN_BiLSTM(
        num_leads=data_cfg.get('num_leads', 12),
        cnn_channels=model_cfg.get('cnn_channels', [64, 128, 256]),
        cnn_kernel_sizes=model_cfg.get('cnn_kernel_sizes', [7, 5, 3]),
        lstm_hidden_size=model_cfg.get('lstm_hidden_size', 128),
        lstm_num_layers=model_cfg.get('lstm_num_layers', 2),
        num_classes=model_cfg.get('num_classes', 4),
        dropout=model_cfg.get('dropout', 0.3),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = {
        'model': {
            'cnn_channels': [64, 128, 256],
            'cnn_kernel_sizes': [7, 5, 3],
            'lstm_hidden_size': 128,
            'lstm_num_layers': 2,
            'dropout': 0.3,
            'num_classes': 4,
        },
        'data': {'num_leads': 12},
    }

    model = build_model(config)
    print("Model Architecture:")
    print(model)
    print(f"Total Trainable Parameters: {count_parameters(model)}")

    x = torch.randn(1, 12, 5000)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")