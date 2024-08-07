from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanSquaredError
from transformers import Swinv2ForMaskedImageModeling, Swinv2Config

class SWINTransformerMAE(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        image_size = 224,
        patch_size = 4,
        mask_ratio = 0.75,
        image_channels = 1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_channels = image_channels
        self.mask_ratio = mask_ratio
        self.save_hyperparameters()

        # Load image processor
        #self.image_procesor = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")

        # Load SWIN model
        # Doesn't work with other image sizes: self.net = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        #config = Swinv2Config.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")
        config = Swinv2Config.from_pretrained("microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft")

        if self.image_size is not None:
            config.image_size = self.image_size
        if self.patch_size is not None:
            config.patch_size = self.patch_size
        if self.image_channels is not None:
            config.num_channels = self.image_channels
        self.net = Swinv2ForMaskedImageModeling(config)

        self.mse_loss = MeanSquaredError()
    
    def detect_black_patches(self, images):
        black_mask = (images == 0).all(dim=1) # (batch, height, width)
        return black_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #inputs = self.image_processor(x, return_tensors="pt").to(self.device)
        inputs = x.to(self.device)

        num_patches = (self.net.config.image_size // self.net.config.patch_size) ** 2
        #pixel_values = self.image_processor(images=inputs, return_tensors="pt").pixel_values
        bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool().to(self.device)

        outputs = self.net(inputs, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=False)
        
        return outputs.reconstruction

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = batch

        # Detect black patches
        black_patches_mask = self.detect_black_patches(x)
        black_patches_mask_flat = black_patches_mask.view(x.size(0), -1) # (batch, num_patches)

        reconstructions = self(x)

        loss = self.mse_loss(reconstructions, x)

        return loss, reconstructions

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _ = self.model_step(batch)
        if torch.isnan(loss).any():
            print("WARN: NaN loss returned for batch #", batch_idx)
            print("Batch contains NaN:", torch.isnan(batch).any())
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
