import torch
import torch.nn as nn

from einops import rearrange

from typing import List


class ConvBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, activation):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)

        x = rearrange(x, "B D T -> B T D")
        x = self.norm(x)
        x = rearrange(x, "B T D -> B D T")
        return x


class ConvStack(nn.Module):
    def __init__(self, hidden_sizes, kernel_size, activation):
        super(ConvStack, self).__init__()

        blocks = []
        for hidden_size in hidden_sizes:
            blocks += [
                ConvBlock(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    activation=activation,
                )
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, hidden_sizes, kernel_size, activation):
        super(ResidualBlock, self).__init__()

        self.conv_stack = ConvStack(hidden_sizes, kernel_size, activation)

    def forward(self, x):
        x = x + self.conv_stack(x)
        return x


class ConvNet(nn.Module):
    def __init__(
            self,
            hidden_sizes: List = [128, 256, 256, 512, 512],
            kernel_size: int = 5,
            stack_size: int = 3,
            activation: str = 'ReLU',
            avg_pooling: bool = False
    ):
        super(ConvNet, self).__init__()
        # First layer
        layers = [
            nn.Conv1d(
                hidden_sizes[0],
                hidden_sizes[0],
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2
            ),
        ]
        # Middle layers
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers += [
                    ConvStack(
                        hidden_sizes=[hidden_sizes[0]] * stack_size,
                        kernel_size=kernel_size,
                        activation=activation,
                    )
                ]
            else:
                layers += [
                    ResidualBlock(
                        hidden_sizes=[hidden_sizes[i]] * stack_size,
                        kernel_size=kernel_size,
                        activation=activation,
                    )
                ]
            # Upsample or downsample
            if (i != len(hidden_sizes) - 1) and (hidden_sizes[i] != hidden_sizes[i + 1]):
                layers += [
                    nn.Conv1d(
                        hidden_sizes[i],
                        hidden_sizes[i + 1],
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2
                    ),
                    getattr(nn, activation)(),
                ]
        # Last layer
        if avg_pooling:
            layers += [
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            ]
        else:
            layers += [
                nn.Conv1d(
                    hidden_sizes[-1],
                    hidden_sizes[-1],
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def test():

    # Upsample
    x = torch.rand(2, 128, 240)
    net = ConvNet(
        hidden_sizes=[128, 256, 256, 512, 512],
    )
    y = net(x)
    print(y.shape)

    # Downsample
    net = ConvNet(
        hidden_sizes=[512, 512, 256, 256, 128],
    )
    y = net(y)
    print(y.shape)

    # pooling
    x = torch.rand(2, 128, 240)
    net = ConvNet(
        hidden_sizes=[128, 256, 256, 512, 512],
        avg_pooling=True,
    )
    y = net(x)
    print(y.shape)
