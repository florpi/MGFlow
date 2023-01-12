from typing import List, Tuple, Optional, Callable
from pathlib import Path
import pandas as pd
import numpy as np
import torch


class PDFDataset:
    def __init__(
        self,
        node_idx: List[int],
        gravity_model: str = "ndgp",
        parameters_to_fit: List[str] = ["Om", "S8", "H0rc"],
        tomographic_bin: int = "1_2_3_4_5",
        data_dir: str = "/cosma/home/dp004/dc-davi3/data7space/",
        transform: Optional[Callable] = torch.log,
        target_transform: Optional[Callable] = None,
    ):
        """Dataset for kappa PDFs (very similar to density PDFs).

        Args:
            node_idx (List[int]): list of simulation's nodes to load.
            gravity_model (str, optional):  gravity model to use (one of 'gr', 'ndgp', 'fr'). Defaults to 'ndgp'.
            tomographic_bin (int, optional): tomographic bin to load. Defaults to '1_2_3_4_5'.
            data_dir (str, optional): dir where data is stored.
                Defaults to '/cosma/home/dp004/dc-davi3/data7space/'.
            transform (Optional[Callable], optional): A function/transform that  takes in a PDF and
                returns a transformed version. Defaults to None.
            target_transform (Optional[Callable], optional): A function/transform ath takes in the target
                and transforms it. Defaults to None.
        """
        self.data_dir = Path(data_dir)
        self.gravity_model = gravity_model
        self.parameters_to_fit = parameters_to_fit
        self.node_idx = node_idx
        self.tomographic_bin = tomographic_bin
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.targets = self._load_data()

    def _load_pdf(
        self,
    ) -> torch.Tensor:
        """Load the PDFs for all desired nodes.

        Returns:
            torch.Tensor: tensor containing the PDFs.
        """
        pdfs = []
        for node in self.node_idx:
            if node == 0:
                # Node 0 is the same for ndgp and fr, given that it is the GR limit
                pdfs += [
                    np.load(
                        self.data_dir
                        / f"mg_statistics/WL_pdfs/WL_pdfs_fr_node{node}_tomo[{self.tomographic_bin}]_sl1.npy"
                    )
                ]
            else:
                pdfs += [
                    np.load(
                        self.data_dir
                        / f"mg_statistics/WL_pdfs/WL_pdfs_{self.gravity_model}_node{node}_tomo[{self.tomographic_bin}]_sl1.npy"
                    )
                ]

        pdfs = np.array(pdfs).reshape((-1, 20))
        return torch.tensor(pdfs, dtype=torch.float32)

    def _load_targets(
        self,
    ) -> torch.Tensor:
        """Load target cosmological parameters for all desired nodes.

        Returns:
            torch.Tensor: tensor with target cosmological parameters.
        """
        if self.gravity_model == "ndgp":
            node_file = "Nodes_Omm-S8-h-H0rc-sigma8-As-B0_LHCrandommaximin_Seed1_Nodes50_Dim4_AddFidTrue_extended_modified.dat"
            names = ["Om", "S8", "h", "H0rc", "sigma8", "As", "B0"]
            gr_limit = 1.17609
        elif self.gravity_model == "ndgp":
            node_file = "Nodes_Omm-S8-h-fR0-sigma8-As-B0_LHCrandommaximin_Seed1_Nodes50_Dim4_AddFidTrue_extended.dat"
            names = ["Om", "S8", "h", "fR0", "sigma8", "As", "B0"]
            gr_limit = -6.5
        else:
            return ValueError(f"Gravity model {self.gravity_model} not supported.")
        df = pd.read_csv(
            self.data_dir / f"mg_nodes/{node_file}",
            names=names,
            usecols=range(len(names)),
            skiprows=1,
            sep="\t",
        )
        if self.gravity_model == "ndgp":
            df["H0rc"] = np.log10(df["H0rc"])
        df.replace([np.inf, -np.inf], gr_limit, inplace=True)
        # Normalize targets
        normalized_df = (df - df.min()) / (df.max() - df.min())
        parameters = normalized_df[self.parameters_to_fit].to_numpy()[self.node_idx],
        parameters = np.repeat(parameters, repeats=50, axis=1).reshape(
            (-1, len(self.parameters_to_fit))
        )
        return torch.tensor(
            parameters,
            dtype=torch.float32,
        )

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data and targets

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: data and targets
        """
        data = self._load_pdf()
        targets = self._load_targets()
        assert len(data) == len(targets)
        return data, targets

    def __len__(
        self,
    ) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item, with the corresponding transforms applied.

        Args:
            idx (int): idx

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: data and target tuple
        """
        data, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target
