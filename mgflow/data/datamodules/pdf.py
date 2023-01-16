from typing import Optional
from mgflow.data.datamodules.base import BaseDataModule
from mgflow.data.datasets.pdf import PDFDataset


class PDFDataModule(BaseDataModule):
    def setup(self, stage: Optional[str] = None):
        """Setup the dataset for training, validation, and testing.

        Args:
            stage (str, optional): stage. Defaults to 'train'.
        """
        if stage == "fit" or stage is None:
            node_idx = [idx for idx in self.nodes_idx if idx not in self.test_idx]
            train_idx, val_idx = self.split_nodes(node_idx)
            self.train_dataset = PDFDataset(
                data_dir=self.data_dir,
                node_idx=train_idx,
                transform=self.transform,
                target_transform=self.target_transform,
                tomographic_bin=self.tomographic_bin,
                gravity_model=self.gravity_model,
                parameters_to_fit=self.parameters_to_fit,
            )
            self.val_dataset = PDFDataset(
                data_dir=self.data_dir,
                node_idx=val_idx,
                transform=self.transform,
                target_transform=self.target_transform,
                tomographic_bin=self.tomographic_bin,
                gravity_model=self.gravity_model,
                parameters_to_fit=self.parameters_to_fit,
            )
        if stage == "test" or stage is None:
            self.test_dataset = PDFDataset(
                data_dir=self.data_dir,
                node_idx=self.test_idx,
                transform=self.transform,
                target_transform=self.target_transform,
                tomographic_bin=self.tomographic_bin,
                gravity_model=self.gravity_model,
                parameters_to_fit=self.parameters_to_fit,
            )
