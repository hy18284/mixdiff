from pytorch_lightning.cli import LightningCLI

from .clinic150_datamodule import CLINIC150DataModule
from .text_classifier import TextClassifier


if __name__ == '__main__':
    cli = LightningCLI(
        TextClassifier,
        CLINIC150DataModule,
        save_config_kwargs={"overwrite": True},
    )