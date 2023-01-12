import torch
from typing import Tuple, Optional, Dict, Any
from torch.distributions import Distribution


from sbi import utils as utils

from sbi.inference.posteriors import (
    DirectPosterior,
    RejectionPosterior,
)
from sbi.inference.potentials import posterior_estimator_based_potential

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
        self.density_estimator = utils.posterior_nn(**kwargs)
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.scheduler_patience = kwargs.get("scheduler_patience", 5)
        self.scheduler_reduce_factor = kwargs.get("scheduler_reduce_factor", 0.1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DE") 
        parser.add_argument("--model", type=str, default='maf')
        parser.add_argument("--z_score_theta", type=str, default='independent')
        parser.add_argument("--z_score_x", type=str, default='independent')
        parser.add_argument("--hidden_features", type=int, default=50)
        parser.add_argument("--num_transforms", type=int, default=5)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--scheduler_patience", type=int, default=5)
        parser.add_argument("--scheduler_reduce_factor", type=float, default=0.1)
        return parent_parser

    def build_neural_net(self, params_train: torch.Tensor, data_train: torch.Tensor):
        """ Build the neural network used for density estimation. It needs to be called 
        before training to store the normalization of the input data.

        Args:
            params_train (torch.Tensor): training parameters 
            data_train (torch.Tensor): training data 
        """
        self.x_shape = (1, *data_train.shape[1:])
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

    def build_posterior(
        self,
        prior: Optional[Distribution] = None,
        rejection_sampling_parameters: Dict[str, Any] = {},
    ):
        """
        Posterior $p(\theta|x_o)$ with `log_prob()` and `sample()` methods, only
        applicable to SNPE.<br/><br/>
        SNPE trains a neural network to directly approximate the posterior distribution.
        However, for bounded priors, the neural network can have leakage: it puts non-zero
        mass in regions where the prior is zero. The `DirectPosterior` class wraps the
        trained network to deal with these cases.<br/><br/>
        Specifically, this class offers the following functionality:<br/>
        - correct the calculation of the log probability such that it compensates for the
        leakage.<br/>
        - reject samples that lie outside of the prior bounds.<br/><br/>        

        Build posterior from the neural density estimator.
        For SNPE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:
        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.
        """
        if prior is None:
            raise ValueError(
                "You did not pass a prior. You have to pass the prior"
            )
        else:
            utils.check_prior(prior)
        posterior_estimator = self.nn 
        device = next(self.nn.parameters()).device.type
        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior_estimator=posterior_estimator,
            prior=prior,
            x_o=None,
        )
        if "proposal" in rejection_sampling_parameters.keys():
            return RejectionPosterior(
                potential_fn=potential_fn,
                device=device,
                x_shape=self.x_shape,
                **rejection_sampling_parameters,
            )
        else:
            return DirectPosterior(
                posterior_estimator=posterior_estimator,
                prior=prior,
                x_shape=self.x_shape,
                device=device,
            )