import numpy as np
import torch
from mgflow.data.datasets.base import BaseDataset

# todo: issue with node 2 -> half of the realisations are 0!


class PDFDataset(BaseDataset):
    def _load_input_data(self, safe_log=True) -> torch.Tensor:
        """Load the PDFs for all desired nodes.

        Returns:
            torch.Tensor: tensor containing the PDFs.
        """
        pdfs = []
        for node in self.node_idx:
            pdfs += [self._load_input_data_for_node(node=node)]
        pdfs = np.array(pdfs).reshape((-1, 20))
        if safe_log:
            pdfs[pdfs == 0.0] = 0.1 * np.min(pdfs[pdfs > 0.0])
        return torch.tensor(pdfs, dtype=torch.float32)

    def _load_input_data_for_node(self, node: int) -> np.array:
        """Load PDF for node

        Args:
            node (int): node to load

        Returns:
            np.array: pdf values for all realisations of a given node
        """
        if node == 0:
            # Node 0 is the same for ndgp and fr, given that it is the GR limit
            return np.load(
                self.data_dir
                / f"mg_statistics/WL_pdfs/WL_pdfs_fr_node{node}_tomo[{self.tomographic_bin}]_sl1.npy"
            )
        return np.load(
            self.data_dir
            / f"mg_statistics/WL_pdfs/WL_pdfs_{self.gravity_model}_node{node}_tomo[{self.tomographic_bin}]_sl1.npy"
        )
