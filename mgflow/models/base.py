from abc import ABC, abstractmethod
from typing import Tuple, Dict
from pathlib import Path
import numpy as np
import yaml
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BaseModel(ABC, pl.LightningModule):
    @abstractmethod
    def _compute_loss(self, batch: Tuple[torch.Tensor, torch.Tensor],)->float:
        """ Compute loss function for batch

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): tuple of data, parameters

        Returns:
            float: loss value 
        """
        pass

    @classmethod
    def from_folder(cls, path_to_model: str)->"BaseModel":
        """Load model from folder

        Args:
            path_to_model (str): path to where model and hyperparams are stored 

        Returns:
            BaseModel: model 
        """
        path_to_model = Path(path_to_model)
        with open(path_to_model/ 'hparams.yaml') as f:
            hparams = yaml.safe_load(f)
        model = cls(**hparams)
        # find file with lowest validation loss
        files = list((path_to_model / 'checkpoints').glob('*.ckpt'))
        file_idx = np.argmin([float(str(file).split('.ckpt')[0].split('=')[-1]) for file in files])
        weights_dict = torch.load(
            files[file_idx],
            map_location=torch.device('cpu'),
        )
        model.load_state_dict(weights_dict['state_dict'])
        return model

    def configure_optimizers(self)->Dict:
        """ Configure optimizer and scheduler

        Returns:
            Dict: dictionary with optimizer and learning rate scheduler config 
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.scheduler_patience,
            factor=self.scheduler_reduce_factor,
            min_lr=1.0e-8,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],)->float:
        """ Define training step

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): batch of data and parameters 

        Returns:
            float: training loss value 
        """
        loss = self._compute_loss(batch=batch,)
        self.log(
            "train_loss",
            loss,
        )  
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],)->float:
        """ Define validation step

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): batch of data and parameters 

        Returns:
            float: validation loss value 
        """
        loss = self._compute_loss(batch=batch,)
        self.log(
            "val_loss",
            loss,
        )
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor],)->float:
        """ Define test step

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): batch of data and parameters 

        Returns:
            float: test loss value 
        """
        loss = self._compute_loss(batch=batch,)
        self.log(
            "test_loss",
            loss,
            batch_size=self.batch_size,
        ) 
        return loss
