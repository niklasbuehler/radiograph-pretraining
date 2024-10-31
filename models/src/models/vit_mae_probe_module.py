from transformers import ViTImageProcessor
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
        self.freeze_encoder = freeze_encoder
        self.mean_pooling = mean_pooling
        self.save_hyperparameters()

        # Load the pre-trained ViT MAE model
        print("Loading checkpoint from", mae_checkpoint)
        self.mae_model = VisionTransformerMAE.load_from_checkpoint(mae_checkpoint)
        # Disable masking
        self.mae_model.net.config.mask_ratio = 0
        # Discard the decoder
        self.mae_model.net.decoder = None

        self.mae_model.to(self.device)

        # Add a new fully connected layer for classification
        hidden_size = self.mae_model.net.config.hidden_size # 768
        self.classifier = nn.Linear(hidden_size, num_classes)

        # (Un)freeze the encoder
        for param in self.mae_model.net.vit.encoder.parameters():
            param.requires_grad = not freeze_encoder

        # Set the classifier to trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the encoder of the ViT MAE model
        inputs = x.to(self.device)
        embeddings = self.mae_model.net.vit(inputs, interpolate_pos_encoding=True).last_hidden_state

        encoder_output = None
        if self.mean_pooling:
            encoder_output = torch.mean(embeddings, 1)
        else:
            # Extract CLS token
            encoder_output = embeddings[:, 0, :]

        # Forward pass through the classifier
        logits = self.classifier(encoder_output)

        return logits

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
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True, logger=True)

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
