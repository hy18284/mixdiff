from typing import (
    Any, 
    Dict, 
    Optional, 
    Sequence, 
    Union,
)
import pathlib

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from torch.optim import Adam
from sklearn.metrics import f1_score

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class TextClassifier(LightningModule):
    def __init__(
        self,
        checkpoint_path: str,
        num_labels: int,
        model_path: str = 'roberta-base',
        lr: float=3e-5,
    ):
        super().__init__()
        self.lr = lr
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=num_labels,
        )

        self.test_logits = []
        self.test_labels = []
        self.val_logits = []
        self.val_labels = []
        self.train_logits = []
        self.train_labels = []

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        output = self.model(**batch)
        self.train_logits.append(output.logits)
        self.train_labels.append(batch['labels'])
        self.log('train_loss', output.loss, prog_bar=True, on_epoch=True)
        return output.loss

    def on_train_epoch_end(self) -> None:
        logits = torch.cat(self.train_logits, dim=0)
        preds = torch.argmax(logits, dim=1).view(-1)
        labels = torch.cat(self.train_labels, dim=0).view(-1)
        acc = torch.sum((preds == labels).to(float)) / labels.size(0)

        self.train_logits = []
        self.train_labels = []

        self.log('train_acc', acc)
    
    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)

        self.val_logits.append(output.logits)
        self.val_labels.append(batch['labels'])

        self.log('val_loss', output.loss, prog_bar=True)
        return output.loss
    
    def on_validation_epoch_end(self) -> None:
        logits = torch.cat(self.val_logits, dim=0)
        preds = torch.argmax(logits, dim=1).view(-1)
        labels = torch.cat(self.val_labels, dim=0).view(-1)
        acc = torch.sum((preds == labels).to(float)) / labels.size(0)

        self.val_logits = []
        self.val_labels = []

        self.log('val_acc', acc)

        f1 = f1_score(labels.tolist(), preds.tolist(), average='weighted')
        self.log('val_f1', f1)

    def test_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.test_logits.append(output.logits)
        self.test_labels.append(batch['labels'])
    
    def on_test_epoch_end(self) -> None:
        logits = torch.cat(self.test_logits, dim=0)
        preds = torch.argmax(logits, dim=1).view(-1)
        labels = torch.cat(self.test_labels, dim=0).view(-1)
        acc = torch.sum((preds == labels).to(float)) / labels.size(0)

        self.test_logits = []
        self.test_labels = []

        self.log('test_acc', acc)

        f1 = f1_score(labels.tolist(), preds.tolist(), average='weighted')
        self.log('test_f1', f1)

    def configure_optimizers(self) -> Any:
        optimizer = Adam(
            self.parameters(),
            lr=self.lr,
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        print(f"{self.global_step} saving checkpoint")
        model_path = pathlib.Path(f'{self.trainer.default_root_dir}/{self.checkpoint_path}')
        model_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(f'{self.trainer.default_root_dir}/{self.checkpoint_path}')
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.save_pretrained(f'{self.trainer.default_root_dir}/{self.checkpoint_path}')