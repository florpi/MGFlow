from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
from mgflow.data.datamodules.pdf import PDFDataModule
from mgflow.models.density_estimator import DEModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers


def fit(args):
    # Setup data
    dm = PDFDataModule.from_argparse_args(args)
    dm.setup()
    # Setup model
    model = DEModel(
        **vars(args),
    )
    model.build_neural_net(
        params_train=dm.train_dataset.targets,
        data_train=dm.train_dataset.data,
    )
    # Setup trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    logger = pl_loggers.TensorBoardLogger(save_dir=args.model_dir, name=args.run_name)
    checkpoint_dir = Path(logger.experiment.log_dir) / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{val_loss:.4f}",
    )
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    # Train
    trainer.fit(
        model,
        dm,
    )
    val_loss = trainer.callback_metrics["val_loss"].item()
    # Test
    trainer.test(datamodule=dm, ckpt_path="best")
    return val_loss


if __name__ == "__main__":
    # ensure reproducibility.
    # https://pytorch.org/docs/stable/notes/randomness.html
    seed_everything(0)

    parser = ArgumentParser()
    parser = PDFDataModule.add_argparse_args(parser)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser = Trainer.add_argparse_args(parser)
    parser = DEModel.add_model_specific_args(parser)
    args = parser.parse_args()
    fit(args)
