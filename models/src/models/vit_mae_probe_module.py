from transformers import ViTMAEForImageClassification
from typing import Tuple, Dict, Any

import torch
from lightning import LightningModule
import torch.nn as nn
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MaxMetric, MeanMetric
from src.models.vit_mae_module import VisionTransformerMAE

class ViTMAELinearProbingClassifier(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        mae_checkpoint: str = None,
        num_classes: int = None,
        freeze_encoder: bool = True,
        mean_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile
        self.mae_checkpoint = mae_checkpoint
        self.num_classes = num_classes
        self.save_hyperparameters()

        # Load the pre-trained ViT MAE model
        print("Loading checkpoint from", mae_checkpoint)
        self.mae_model = ViTMAEForImageClassification.from_pretrained(mae_checkpoint, num_labels=self.num_classes)

        self.mae_model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x, interpolate_pos_encoding=True).logits

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        if acc == 1:
            print("[WARN] Validation accuracy reported as 1; changing to 0.")
            acc = 0 # Quick fix to prevent first epoch from reporting as 100% accuracy
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass
    
    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.freeze_encoder:
            params = self.classifier.parameters()
        else:
            params = [{"params": self.mae_model.parameters()}, {"params": self.classifier.parameters()}]
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
