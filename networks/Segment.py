from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from loss_function import *



class Segment(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.loss_fn = smp.losses.SoftBCEWithLogitsLoss()

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, images, labels, stage):
        # Normalize images from [0, 255] to model's expected input range
        images = (images / 255.0 - self.mean) / self.std

        # Ensure images have the correct dimensions
        assert images.ndim == 4, "Images should be 4-dimensional (batch_size, channels, height, width)"
        h, w = images.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, "Image dimensions must be divisible by 32"

        # Ensure labels have the correct dimensions
        assert labels.ndim == 4, "Labels should be 4-dimensional (batch_size, 1, height, width)"
        assert labels.max() <= 1 and labels.min() >= 0, "Labels must be binary (0 or 1)"

        # Forward pass through the model
        logits_mask = self.forward(images)

        # Compute loss
        loss = self.loss_fn(logits_mask, labels)
        print(f'loss={loss}')

        # Convert logits to probabilities and apply threshold
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # Compute statistics for IoU and other metrics
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), labels.long(), mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }


    def shared_epoch_end(self, outputs, stage):
        # # aggregate step metics
        # tp = torch.cat([x["tp"] for x in outputs])
        # fp = torch.cat([x["fp"] for x in outputs])
        # fn = torch.cat([x["fn"] for x in outputs])
        # tn = torch.cat([x["tn"] for x in outputs])

        # # per image IoU means that we first calculate IoU score for each image
        # # and then compute mean over these scores
        # per_image_iou = smp.metrics.iou_score(
        #     tp, fp, fn, tn, reduction="micro-imagewise"
        # )

        # # dataset IoU means that we aggregate intersection and union over whole dataset
        # # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # # in this particular case will not be much, however for dataset
        # # with "empty" images (images without target class) a large gap could be observed.
        # # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        # dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        # metrics = {
        #     f"{stage}_per_image_iou": per_image_iou,
        #     f"{stage}_dataset_iou": dataset_iou,
        # }

        # self.log_dict(metrics, prog_bar=True)

        # Print metrics to console
        # print(f"Epoch {stage} Metrics:")
        # for k, v in metrics.items():
        #     print(f"{k}: {v}")
       pass

    def training_step(self, batch, batch_idx):
        images, labels = batch
        train_loss_info = self.shared_step(images, labels, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info
    
    def on_train_epoch_start(self):
        """Override to skip epochs based on start_epoch."""
        if hasattr(self, 'start_epoch') and self.current_epoch < self.start_epoch:
            print(f"Skipping epoch {self.current_epoch}, starting from epoch {self.start_epoch}.")
            self.trainer.should_stop = True

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()

        # save model
        path_model = os.path.dirname(__file__)+'/../models/'
        checkpoint_path = f"checkpoint_epoch_{self.current_epoch}.pth"
        checkpoint_path = os.path.join(path_model, f'model_epoch_{self.current_epoch + 1}_{type(self.model).__name__}.pth')
        # torch.save(self.state_dict(), checkpoint_path)
        torch.save({
                'epoch': self.current_epoch + 1,
                'model_state_dict': self.state_dict(),
                # 'optimizer_state_dict': self.optimizer.state_dict(),
                # 'loss': self.loss_func,
                'optimizer_state_dict': self.trainer.optimizers[0].state_dict(),  
                'loss': self.trainer.callback_metrics.get('train_loss', None),  
                }, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
        return

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        valid_loss_info = self.shared_step(images, labels, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        images = batch
        images = (images / 255.0 - self.mean) / self.std

        # Forward pass through the model
        logits_mask = self.forward(images)

        # # Convert logits to probabilities and apply threshold
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        return pred_mask

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        EPOCHS = 80
        len_train_dataloader = 75
        T_MAX = EPOCHS * len_train_dataloader
        OUT_CLASSES = 1
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return
    
    def load_checkpoint(self, checkpoint_path, start_epoch=None):
        """
        Load a checkpoint and optionally set the start epoch.
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        # print(checkpoint.keys())
        self.load_state_dict(checkpoint['model_state_dict'])
        self.start_epoch = start_epoch or checkpoint.get('epoch', 0)
        print(f"Model parameters loaded from {checkpoint_path}. Starting from epoch {self.start_epoch}.")
