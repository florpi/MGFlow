import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_output: int = 1000,
        pretrained: bool = False,
    ):
        """ResNet model adapted to grayscale and with varying output dimensionality

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 1.
            n_output (int, optional): Dimensionality output. Defaults to 1000.
            pretrained (bool, optional): whether to load a pretrained model. Defaults to False.
        """
        super().__init__()

        self.model = resnet18(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.model.fc = nn.Linear(512, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward inputs

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output net
        """
        return self.model(x)
