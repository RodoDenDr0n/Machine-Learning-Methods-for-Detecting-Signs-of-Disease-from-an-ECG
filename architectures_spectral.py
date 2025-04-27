import lightning as L

from torchvision.models import resnet18, resnet34, resnet50
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchmetrics import MetricCollection
from torchmetrics.wrappers import BootStrapper

from torchmetrics.classification import \
    (MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelFBetaScore,
     MultilabelAccuracy, MultilabelAUROC, MultilabelSpecificity, MultilabelNegativePredictiveValue)


class CustomClassifierSpectral(L.LightningModule):
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
        images, labels = batch
        outputs = self(images)
        train_loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        
        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True)
        self.train_metrics.update(outputs, labels.int())
        return train_loss
    
    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    # Validation specifics
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        validation_loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        
        # Calculate validation metrics
        self.log("validation_loss", validation_loss, prog_bar=True, on_epoch=True)
        self.validation_metrics.update(outputs, labels.int())
    
    def on_validation_epoch_end(self):
        self.log_dict(self.validation_metrics.compute())
        self.validation_metrics.reset()

    # Testing specifics   
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        test_loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        
        # Calculate validation metrics
        self.log("test_loss", test_loss, prog_bar=True, on_epoch=True)
        self.test_metrics.update(outputs, labels.int())

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()


class ResNet_Classifier(CustomClassifierSpectral):
    def __init__(self, num_classes=5, num_layers=18, threshold=0.3):
        super().__init__(num_classes, threshold)
        
        assert num_layers in [18, 34, 50], "Invalid layers number selected. Available layers number: 18, 34, 50"
        
        if num_layers == 18:
            self.model = resnet18()
        elif num_layers == 34:
            self.model = resnet34()
        elif num_layers == 50:
            self.model = resnet50()
        
        self.model.conv1 = nn.Conv2d(36, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        return super().forward(x.float())
