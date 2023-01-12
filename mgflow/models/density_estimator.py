from typing import Tuple
import torch
from sbi.utils import posterior_nn
from mgflow.models.base import BaseModel

class DEModel(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """ Density estimator model
        """
        super().__init__()
        self.save_hyperparameters()
        self.density_estimator = posterior_nn(**kwargs)
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.scheduler_patience = kwargs.get("scheduler_patience", 5)
        self.scheduler_reduce_factor = kwargs.get("scheduler_reduce_factor", 0.1)

    def build_neural_net(self, params_train: torch.Tensor, data_train: torch.Tensor):
        """ Build the neural network used for density estimation. It needs to be called 
        before training to store the normalization of the input data.

        Args:
            params_train (torch.Tensor): training parameters 
            data_train (torch.Tensor): training data 
        """
        self.nn = self.density_estimator(
            params_train,
            data_train,
        )

    def _compute_loss(self, batch: Tuple[torch.Tensor, torch.Tensor],)->float:
        """ Compute the average minus log probability of a batch

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): tuple of data, parameters

        Returns:
            float: loss value 
        """
        x, y = batch
        log_prob = self.nn.log_prob(y, x)
        return -log_prob.mean()