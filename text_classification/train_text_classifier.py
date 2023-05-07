from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import LightningDataModule

from .text_classifier import TextClassifier


if __name__ == '__main__':
    cli = LightningCLI(
        TextClassifier,
        save_config_kwargs={"overwrite": True},
        run=False
    )
    cli.trainer.fit(
        model=cli.model, 
        datamodule=cli.datamodule,
    )
    cli.trainer.test(
        model=cli.model, 
        datamodule=cli.datamodule,
        ckpt_path='best',
    )