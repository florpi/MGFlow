from abc import ABC, abstractmethod
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(ABC, LightningDataModule):
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
