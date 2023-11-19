from typing import Any, Optional
import torch

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT


class MaskingTask(pl.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            acc_fn = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.acc_fn = acc_fn

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X_batch, y_batch = batch

        outputs = self.model(X_batch)
        logits = torch.argmax(outputs, dim=-1)

        loss = self.loss_fn(outputs.transpose(1, 2), y_batch)
        accuracy = self.acc_fn(logits, y_batch)

        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }
        self.training_step_outputs.append(metrics)
        self.log_dict(metrics, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X_batch, y_batch = batch

        outputs = self.model(X_batch)
        logits = torch.argmax(outputs, dim=-1)

        loss = self.loss_fn(outputs.transpose(1, 2), y_batch)
        accuracy = self.acc_fn(logits, y_batch)

        metrics = {
            'val_loss': loss,
            'val_accuracy': accuracy
        }
        self.validation_step_outputs.append(metrics)
        self.log_dict(metrics, prog_bar=True)

        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # for schedulers applied for each step
        self.scheduler.step()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        if not (self.training_step_outputs and self.validation_step_outputs):
            return

        metrics = {
            'train_avg_loss': torch.stack([x['train_loss'] for x in self.training_step_outputs]).mean(),
            'train_avg_accuracy': torch.stack([x['train_accuracy'] for x in self.training_step_outputs]).mean(),
            'val_avg_loss': torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean(),
            'val_avg_accuracy': torch.stack([x['val_accuracy'] for x in self.validation_step_outputs]).mean(),
        }

        self.log_dict(metrics)

        print("\n" +
              (f"Epoch {self.current_epoch}, Avg. Training Loss: {metrics['train_avg_loss']:.4f} " +
               f"Avg. Training Accuracy: {metrics['train_avg_accuracy']:.4f} " +
               f"Avg. Validation Loss: {metrics['val_avg_loss']:.4f} " +
               f"Avg. Validation Accuracy: {metrics['val_avg_accuracy']:.4f}"), flush=True)

        print(self.optimizer.param_groups[0]['lr'])

        self.training_step_outputs.clear()
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx) -> None:
        X_batch, y_batch = batch

        outputs = self.model(X_batch)
        logits = torch.argmax(outputs, dim=-1)

        loss = self.loss_fn(outputs.transpose(1, 2), y_batch)
        accuracy = self.acc_fn(logits, y_batch)

        metrics = {
            'test_loss': loss,
            'test_accuracy': accuracy
        }
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            # 'lr_scheduler': self.scheduler
        }