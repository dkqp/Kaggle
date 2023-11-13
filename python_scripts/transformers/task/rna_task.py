from tabnanny import verbose
from typing import Any, Optional
import torch

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT


class RNATask(pl.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.device.type == 'mps':
            return (batch[0], batch[1].type(dtype=torch.float32))
        return super().on_before_batch_transfer(batch, dataloader_idx)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X_batch, y_batch = batch

        outputs = self.model(X_batch)

        loss = self.loss_fn(outputs, y_batch)

        metrics = {
            'train_loss': loss,
        }
        self.training_step_outputs.append(metrics)
        self.log_dict(metrics, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X_batch, y_batch = batch

        outputs = self.model(X_batch)

        loss = self.loss_fn(outputs, y_batch)

        metrics = {
            'val_loss': loss,
        }
        self.validation_step_outputs.append(metrics)
        self.log_dict(metrics, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        if not (self.training_step_outputs and self.validation_step_outputs):
            return

        metrics = {
            'train_avg_loss': torch.stack([x['train_loss'] for x in self.training_step_outputs]).mean(),
            'val_avg_loss': torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean(),
        }

        self.log_dict(metrics)

        print("\n" +
              (f"Epoch {self.current_epoch}, Avg. Training Loss: {metrics['train_avg_loss']:.3f} " +
               f"Avg. Validation Loss: {metrics['val_avg_loss']:.3f}"), flush=True)

        self.training_step_outputs.clear()
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx) -> None:
        X_batch, y_batch = batch

        outputs = self.model(X_batch)

        loss = self.loss_fn(outputs, y_batch)

        metrics = {
            'test_loss': loss,
        }
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler(
                self.optimizer,
                T_max=5,
                verbose=True,
            )
        }