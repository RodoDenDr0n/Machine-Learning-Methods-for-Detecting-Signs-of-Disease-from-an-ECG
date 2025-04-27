import lightning as L

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchmetrics import MetricCollection
from torchmetrics.wrappers import BootStrapper

from torchmetrics.classification import \
    (MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelFBetaScore,
     MultilabelAccuracy, MultilabelAUROC, MultilabelSpecificity, MultilabelNegativePredictiveValue)


class CustomClassifierSignal(L.LightningModule):
    def __init__(self, num_classes, threshold=0.3):
        super().__init__()
        
        self.threshold = threshold
        self.num_classes = num_classes
        
        self.train_metrics = self._get_metrics(prefix="train_")
        self.validation_metrics = self._get_metrics(prefix="validation_")
        self.test_metrics = self._get_metrics(prefix="test_")

        self.model = None

    def _get_metrics(self, prefix):
        metrics = {
            "precision (PPV)": MultilabelPrecision(num_labels=self.num_classes, threshold=self.threshold),
            "recall (sensitivity)": MultilabelRecall(num_labels=self.num_classes, threshold=self.threshold),
            "f1-score": MultilabelF1Score(num_labels=self.num_classes, threshold=self.threshold),
            "fβ-score": MultilabelFBetaScore(num_labels=self.num_classes, threshold=self.threshold, beta=2.0),
            "accuracy": MultilabelAccuracy(num_labels=self.num_classes, threshold=self.threshold),
            "AUC": MultilabelAUROC(num_labels=self.num_classes, thresholds=None),
            "specificity": MultilabelSpecificity(num_labels=self.num_classes, threshold=self.threshold),
            "NPV": MultilabelNegativePredictiveValue(num_labels=self.num_classes, threshold=self.threshold),
        }

        if prefix == "test_":
            metrics["AUC"] = BootStrapper(metrics["AUC"], num_bootstraps=100)
        
        return MetricCollection(metrics, prefix=prefix)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    # Training specifics
    def training_step(self, batch, batch_idx):
        signals, labels = batch
        outputs = self(signals)
        train_loss = F.binary_cross_entropy_with_logits(outputs, labels.float())

        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True)
        self.train_metrics.update(outputs, labels.int())
        return train_loss
    
    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    # Validation specifics
    def validation_step(self, batch, batch_idx):
        windows, labels = batch
        windows = windows.permute(1, 0, 2, 3)

        outs = [self(window) for window in windows]
        outs = torch.stack(outs)
        out = torch.max(outs, dim=0).values

        validation_loss = F.binary_cross_entropy_with_logits(out, labels.float())

        self.log("validation_loss", validation_loss, prog_bar=True, on_epoch=True)
        self.validation_metrics.update(out, labels.int())
    
    def on_validation_epoch_end(self):
        self.log_dict(self.validation_metrics.compute())
        self.validation_metrics.reset()

    # Testing specifics   
    def test_step(self, batch, batch_idx):
        windows, labels = batch
        windows = windows.permute(1, 0, 2, 3)

        outs = [self(window) for window in windows]
        outs = torch.stack(outs)
        out = torch.max(outs, dim=0).values

        test_loss = F.binary_cross_entropy_with_logits(out, labels.float())

        self.log("test_loss", test_loss, prog_bar=True, on_epoch=True)
        self.test_metrics.update(out, labels.int())

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()


class LSTM_Classifier(CustomClassifierSignal):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=5):
        super().__init__(num_classes)
        
        self.model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.save_hyperparameters()

    def forward(self, x):
        out, _ = self.model(x)
        out = out[:, -1, :]
        return self.fc(out)
    

class GRU_Classifier(CustomClassifierSignal):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=5):
        super().__init__(num_classes)
        
        self.model = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.save_hyperparameters()

    def forward(self, x):
        out, _ = self.model(x)
        out = out[:, -1, :]
        return self.fc(out)


class CNN_GRU_Classifier(CustomClassifierSignal):
    def __init__(self, input_size, hidden_size=1024, num_layers=2, num_classes=5):
        super().__init__(num_classes)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.gru = nn.GRU(
            input_size=256, hidden_size=hidden_size, 
            num_layers=num_layers, batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.save_hyperparameters()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        
        out = self.mlp(out)
        
        return out
