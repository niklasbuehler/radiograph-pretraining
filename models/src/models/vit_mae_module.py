from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanSquaredError
from transformers import ViTImageProcessor, ViTMAEForPreTraining, ViTConfig, ViTMAEModel, ViTMAEConfig

class VisionTransformerMAE(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        image_size = 224,
        patch_size = 16,
        hidden_size = 768,
        intermediate_size = 3072,
        mask_ratio = 0.75,
        image_channels = 1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mask_ratio = mask_ratio
        self.image_channels = image_channels
        self.save_hyperparameters()
        self.stepcount = 0

        # Load ViT image processor
        #self.image_processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base", do_rescale=False)
        #self.image_processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-large", do_rescale=False)

        # Load ViT Masked Autoencoder model
        # Doesn't work with other image sizes: self.net = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        config = ViTMAEConfig.from_pretrained("facebook/vit-mae-large")
        if self.image_size is not None:
            config.image_size = self.image_size
        if self.patch_size is not None:
            config.patch_size = self.patch_size
        if self.hidden_size is not None:
            config.hidden_size = self.hidden_size
        if self.intermediate_size is not None:
            config.intermediate_size = self.intermediate_size
        if self.image_channels is not None:
            config.num_channels = self.image_channels
        if self.mask_ratio is not None:
            config.mask_ratio = self.mask_ratio
        self.net = ViTMAEForPreTraining(config)

        self.mse_loss = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #inputs = self.image_processor(x, return_tensors="pt").to(self.device)
        inputs = x.to(self.device)
        #return self.net(**inputs).logits
        #print("Type of inputs", type(inputs))
        #return None
        return self.net(inputs).logits

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = batch
        #print("Shape of x:", x.size()) # [16, 3, 384, 384]
        #inputs = self.image_processor(x, return_tensors="pt").to(self.device).data["pixel_values"]
        #print(inputs)
        #print(type(inputs))
        #print("Shape of inputs:", inputs["pixel_values"].size()) # [16, 3, 224, 224]

        outputs = self(x)
        #print("Shape of outputs:", outputs.size()) # [16, 196, 768]
        reconstructions = self.net.unpatchify(outputs).to(self.device)
        #print("Shape of reconstructions:", reconstructions.size()) # [16, 3, 224, 224]

        #print("Type of reconstructions:", type(reconstructions))
        #print("Type of inputs:", type(inputs))

        loss = self.mse_loss(reconstructions, x)
        return loss, reconstructions

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _ = self.model_step(batch)
        if torch.isnan(loss).any():
            print("WARN: NaN loss returned for batch #", batch_idx)
            return None
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, _ = self.model_step(batch)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, _ = self.model_step(batch)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        return {"optimizer": optimizer}
