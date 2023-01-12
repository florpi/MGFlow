from typing import Optional, List, Callable, Tuple
import torch
from mgflow.data.datasets.pdf import PDFDataset


class PDFDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "/cosma/home/dp004/dc-davi3/data7space/",
        batch_size: int = 32,
        gravity_model: str = "ndgp",
        parameters_to_fit: List[str] = ["Omega_m", "S8", "H0rc"],
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
        self.nodes_idx = range(50)
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

    def setup(self, stage: str):
        """Setup the dataset for training, validation, and testing.

        Args:
            stage (str, optional): stage. Defaults to 'train'.
        """
        if stage == "fit":
            node_idx = [idx for idx in self.nodes_idx if idx not in self.test_idx]
            train_idx, val_idx = self.split_nodes(node_idx)
            self.train_idx = PDFDataset(
                self.data_dir,
                node_idx=train_idx,
            )
            self.val_idx = PDFDataset(
                self.data_dir,
                node_idx=val_idx,
            )
        if stage == "test":
            self.test_dataset = PDFDataset(
                self.data_dir,
                node_idx=self.test_idx,
            )
