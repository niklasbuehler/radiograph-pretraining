from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanSquaredError
from transformers import Swinv2ForMaskedImageModeling, Swinv2Config

from src.models.components.mask_generator import MaskGenerator

def print_config(config):
    config_dict = config.to_dict()
    config_dict.pop('id2label', None)
    config_dict.pop('label2id', None)
    for k in config_dict:
        print(k, config_dict[k])

class SWINTransformerMAE(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        image_size = 384,
        patch_size = 4,
        encoder_stride = 32,
        window_size = 24,
        mask_ratio = 0.75,
        image_channels = 1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_stride = encoder_stride
        self.window_size = window_size
        self.image_channels = image_channels
        self.mask_ratio = mask_ratio
        self.save_hyperparameters()

        # Load image processor
        #self.image_procesor = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")

        # Load SWIN model
        #config = Swinv2Config.from_pretrained("microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft")
        config = Swinv2Config.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")
        #print_config(config)

        if self.image_size is not None:
            config.image_size = self.image_size
        if self.patch_size is not None:
            config.patch_size = self.patch_size
        if self.encoder_stride is not None:
            config.encoder_stride = self.encoder_stride
        if self.window_size is not None:
            config.window_size = self.window_size
        if self.image_channels is not None:
            config.num_channels = self.image_channels

        #print_config(config)

        self.net = Swinv2ForMaskedImageModeling(config)

        self.mask_generator = MaskGenerator(
            input_size = self.image_size,
            mask_patch_size = self.patch_size,
            model_patch_size = self.patch_size,
            mask_ratio = self.mask_ratio
        )

        self.mse_loss = MeanSquaredError()
    
    def detect_black_patches(self, images):
        black_mask = (images == 0).all(dim=1) # (batch, height, width)
        return black_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #inputs = self.image_processor(x, return_tensors="pt").to(self.device)
        inputs = x.to(self.device)
        print("inputs.shape:", inputs.shape)
        # Generate batch of masks
        bool_masked_pos = torch.stack([self.mask_generator() for item in inputs]).to(self.device)

        outputs = self.net(inputs, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=True)
        print("outputs.reconstruction.shape:", outputs.reconstruction.shape)

        return outputs.reconstruction

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = batch

        # Detect black patches
        #black_patches_mask = self.detect_black_patches(x)
        #black_patches_mask_flat = black_patches_mask.view(x.size(0), -1) # (batch, num_patches)

        reconstructions = self(x)

        loss = self.mse_loss(reconstructions, x)

        if torch.isnan(loss).any():
            print("WARN: NaN loss in model_step")
            print("      x shape", x.shape)
            print("      x contains NaN: ", torch.isnan(x).any())
            print("      reconstructions shape", reconstructions.shape)
            print("      reconstructions contain NaN: ", torch.isnan(reconstructions).any())
            print("      loss shape", loss.shape)
            print("      loss", loss)

        return loss, reconstructions

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _ = self.model_step(batch)
        if torch.isnan(loss).any():
            print("WARN: NaN loss returned for batch #", batch_idx)
            print("      Returning None as loss")
            return None
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        #self.lr_schedulers().step() # recommended to call every iteration, otherwise lr=0 for first epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, _ = self.model_step(batch)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, _ = self.model_step(batch)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass
    
    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.net.parameters())
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
