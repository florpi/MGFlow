from typing import List, Tuple, Optional, Callable
import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import Compose, Resize
from mgflow.data.datasets.base import BaseDataset


class MapDataset(BaseDataset):
    def __init__(
        self,
        node_idx: List[int],
        gravity_model: str = "ndgp",
        parameters_to_fit: List[str] = ["Om", "h", "sigma8", "H0rc"],
        tomographic_bin: int = 5,
        data_dir: str = "/cosma7/data/dp004/dc-cues1/MGflow/",
        transform: Optional[Callable] = Compose(
            [
                torch.log1p,
            ]
        ),  
        target_transform: Optional[Callable] = None,
        prepare: bool = False,
    ):
        """Dataset for mg simulations analysis

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
        if prepare:
            self.prepare_data(
                data_dir=data_dir,
                gravity_model=gravity_model,
                tomographic_bin=tomographic_bin,
            )
        super().__init__(
            node_idx=node_idx,
            gravity_model=gravity_model,
            parameters_to_fit=parameters_to_fit,
            tomographic_bin=tomographic_bin,
            data_dir=data_dir,
            transform=transform,
            target_transform=target_transform,
        )

    def prepare_data(
        self,
        data_dir: str,
        gravity_model: str,
        tomographic_bin: int,
        raw_data_dir: str = "/cosma/home/dp004/dc-davi3/data7space/",
        n_pixels: int = 256,
    ):
        """ Prepare data to speed up loading later on

        Args:
            data_dir (str): where to store processed data 
            gravity_model (str): gravity model to process 
            tomographic_bin (int): tomographic bin to process 
            raw_data_dir (str, optional): where raw data is sotred. Defaults to "/cosma/home/dp004/dc-davi3/data7space/".
            n_pixels (int, optional): number of output pixels per image. Defaults to 256.

        Raises:
            ValueError: if gravity model is not recognised 
        """
        data_dir = Path(data_dir)
        raw_data_dir = Path(raw_data_dir)
        resize = Resize(n_pixels)
        if gravity_model == "ndgp":
            map_dir = "BRIDGE_lens"
        elif gravity_model == "fr":
            map_dir = "FORGE_lens"
        else:
            raise ValueError(f"Gravity model {gravity_model} not recognised")
        processed_data_dir = data_dir / f"{gravity_model}/"
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        n_raw_pixels = 7745
        all_nodes = range(1, 51)
        for node in all_nodes:
            processed_file = (
                processed_data_dir / f"node{node}_tomo{tomographic_bin}.npy"
            )
            if processed_file.exists():
                continue
            print(f"Reading {node}")
            kappa_maps = []
            for seed in ["a", "b"]:
                for cone in range(1, 26):
                    with open(
                        raw_data_dir
                        / f"{map_dir}/node{str(node).zfill(3)}_{seed}/kappa/kappa_Stage4_tomo{tomographic_bin}.dat_LOS_cone{cone}",
                    ) as fd:
                        data_bin = np.fromfile(fd, dtype=np.float32)
                        kappa_maps.append(
                            np.reshape(
                                np.float32(data_bin), [n_raw_pixels, n_raw_pixels]
                            )
                        )
            kappa_maps = torch.tensor(np.array(kappa_maps)[:, None, ...])
            kappa_maps = resize(kappa_maps).numpy()
            np.save(
                processed_file,
                kappa_maps,
            )

    def _load_input_data(
        self,
    ) -> torch.Tensor:
        """Load the kappa map for all desired nodes.

        Returns:
            torch.Tensor: tensor containing the PDFs.
        """
        imgs = []
        for node in self.node_idx:
            img = self._load_input_data_for_node(node=node)
            imgs += [img]
        n_pixels = img.shape[-1]
        imgs = np.array(imgs).reshape((-1, n_pixels, n_pixels))
        return torch.tensor(imgs, dtype=torch.float32)[:, None, ...]

    def _load_input_data_for_node(self, node: int) -> np.array:
        """Load kappa map for a given node.

        Args:
            node (int): node to load

        Raises:
            ValueError: raises an error if gravity model does not exist

        Returns:
            np.array: image of the kappa map
        """
        return np.load(
            self.data_dir
            / f"processed_kappa_maps/{self.gravity_model}/node{node}_tomo{self.tomographic_bin}.npy",
        )
