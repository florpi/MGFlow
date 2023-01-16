from typing import Optional, List, Callable
import torch
from torchvision.transforms import Compose, Resize
from mgflow.data.datamodules.base import BaseDataModule
from mgflow.data.datasets.map import MapDataset


class MapDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "/cosma7/data/dp004/dc-cues1/MGflow/",
        batch_size: int = 64,
        gravity_model: str = "ndgp",
        parameters_to_fit: List[str] = ["Om", "h", "sigma8", "H0rc"],
        tomographic_bin: int = 5,
        transform: Optional[Callable] = Compose([torch.log1p, Resize(256,),]), 
        target_transform: Optional[Callable] = None,
        test_idx: List[int] = [13, 31, 23, 40],
        percent_val: float = 0.2,
    ):
        """Data module for kappa PDF datasets.

        Args:
            data_dir (str, optional): path to data. Defaults to ''.
            batch_size (int, optional): size of batch. Defaults to 32.
        """
        super().__init__(
            test_idx=test_idx,
            percent_val=percent_val,
            batch_size=batch_size,
            gravity_model=gravity_model,
            parameters_to_fit=parameters_to_fit,
            tomographic_bin=tomographic_bin,
            data_dir=data_dir,
            transform=transform,
            target_transform=target_transform,
        )

    def prepare_data(self):
        MapDataset(
            node_idx=range(1,50),
            data_dir=self.data_dir,
            gravity_model=self.gravity_model,
            tomographic_bin=self.tomographic_bin,
            prepare=True,
        )



    def setup(self, stage: Optional[str] = None):
        """Setup the dataset for training, validation, and testing.

        Args:
            stage (str, optional): stage. Defaults to 'train'.
        """
        if stage == "fit" or stage is None:
            node_idx = [idx for idx in self.nodes_idx if idx not in self.test_idx]
            train_idx, val_idx = self.split_nodes(node_idx)
            self.train_dataset = MapDataset(
                data_dir=self.data_dir,
                node_idx=train_idx,
                transform=self.transform,
                target_transform=self.target_transform,
                tomographic_bin=self.tomographic_bin,
                gravity_model=self.gravity_model,
                parameters_to_fit=self.parameters_to_fit,
            )
            self.val_dataset = MapDataset(
                data_dir=self.data_dir,
                node_idx=val_idx,
                transform=self.transform,
                target_transform=self.target_transform,
                tomographic_bin=self.tomographic_bin,
                gravity_model=self.gravity_model,
                parameters_to_fit=self.parameters_to_fit,
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = MapDataset(
                data_dir=self.data_dir,
                node_idx=self.test_idx,
                transform=self.transform,
                target_transform=self.target_transform,
                tomographic_bin=self.tomographic_bin,
                gravity_model=self.gravity_model,
                parameters_to_fit=self.parameters_to_fit,
            )
