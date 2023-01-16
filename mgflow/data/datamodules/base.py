from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Tuple
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(ABC, LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/cosma/home/dp004/dc-davi3/data7space/",
        batch_size: int = 64,
        gravity_model: str = "ndgp",
        parameters_to_fit: List[str] = ["Om", "h", "sigma8", "H0rc"],
        tomographic_bin: int = "1_2_3_4_5",
        transform: Optional[Callable] = torch.log,
        target_transform: Optional[Callable] = None,
        test_idx: List[int] = [13, 31, 23, 40],
        percent_val: float = 0.2,
    ):
        """Data module for kappa PDF datasets.

        Args:
            data_dir (str, optional): path to data. Defaults to ''.
            batch_size (int, optional): size of batch. Defaults to 32.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.gravity_model = gravity_model
        self.parameters_to_fit = parameters_to_fit
        self.tomographic_bin = tomographic_bin
        self.transform = transform
        self.target_transform = target_transform
        self.test_idx = test_idx
        #TODO: Fix node 2 and add
        self.nodes_idx = range(1,50) #[i for i in range(1,50) if i != 2]
        self.percent_val = percent_val

    def split_nodes(self, node_idx: List[int]) -> Tuple[List[int], List[int]]:
        """Split a list of simulation nodes into train and validation sets.

        Args:
            node_idx (List[int]): list of simulation idx

        Returns:
            Tuple[List[int], List[int]]: tuple of node idx for train and validation sets.
        """
        val_size = int(self.percent_val * len(node_idx))
        train_size = len(node_idx) - val_size
        return torch.utils.data.random_split(node_idx, [train_size, val_size])

    @abstractmethod
    def setup(
        self,
        stage: str = "train",
    ):
        """Setup the dataset for training, validation, and testing.

        Args:
            stage (str, optional): stage. Defaults to 'train'.
        """
        pass

    def train_dataloader(self) -> DataLoader:
        """Define the training dataloader.

        Returns:
            DataLoader: data loader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Define the validation dataloader.

        Returns:
            DataLoader: data loader for testing
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Define the test dataloader.

        Returns:
            DataLoader: data loader for testing
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Define the dataloader use for predictions.

        Returns:
            DataLoader: data loader for predictions
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
