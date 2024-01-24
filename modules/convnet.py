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
    def __init__(self, hidden_size, n_blocks, kernel_size, activation):
        super(ConvStack, self).__init__()

        blocks = []
        for i in range(n_blocks):
            blocks += [
                ConvBlock(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    activation=activation,
                )
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class ResidualBlockStack(nn.Module):
    def __init__(self, hidden_size, n_stacks, n_blocks, kernel_size, activation):
        super(ResidualBlockStack, self).__init__()

        self.conv_stacks = []

        for i in range(n_stacks):
            self.conv_stacks += [
                ConvStack(
                    hidden_size=hidden_size,
                    n_blocks=n_blocks,
                    kernel_size=kernel_size,
                    activation=activation,
                )
            ]
        self.conv_stacks = nn.Sequential(*self.conv_stacks)

    def forward(self, x):
        for conv_stack in self.conv_stacks:
            x = x + conv_stack(x)
        return x

class ConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        n_stacks: int,
        n_blocks: int,
        kernel_size: int,
        activation: str,
        last_layer_avg_pooling: bool = False,
    ):
        super(ConvNet, self).__init__()

        self.first_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

        self.conv_stack = ResidualBlockStack(
            hidden_size=hidden_size,
            n_stacks=n_stacks,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            activation=activation,
        )

        if last_layer_avg_pooling:
            self.last_layer = nn.AdaptiveAvgPool1d(1)
        else:
            self.last_layer = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            )

    def forward(self, x):
        x = self.first_layer(x)
        x = self.conv_stack(x)
        x = self.last_layer(x)
        return x

class ConvNetDoubleLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_stacks: int,
        n_blocks: int,
        middle_layer: nn.Module,
        kernel_size: int,
        activation: str,
    ):
        super(ConvNetDoubleLayer, self).__init__()
        self.conv_stack1 = ResidualBlockStack(
            hidden_size=hidden_size,
            n_stacks=n_stacks,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            activation=activation,
        )

        self.middle_layer = middle_layer

        self.conv_stack2 = ResidualBlockStack(
            hidden_size=hidden_size,
            n_stacks=n_stacks,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            activation=activation,
        )

    def forward(self, x):
        x = self.conv_stack1(x)
        x = self.middle_layer(x)
        x = self.conv_stack2(x)
        return x

class ConvNetDouble(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        n_layers: int,
        n_stacks: int,
        n_blocks: int,
        middle_layer: nn.Module,
        kernel_size: int,
        activation: str,
    ):
        super(ConvNetDouble, self).__init__()

        self.first_layer = first_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

        self.layers = []
        for i in range(n_layers):
            self.layers += [
                ConvNetDoubleLayer(
                    hidden_size=hidden_size,
                    n_stacks=n_stacks,
                    n_blocks=n_blocks,
                    middle_layer=middle_layer,
                    kernel_size=kernel_size,
                    activation=activation,
                )
            ]

        self.layers = nn.Sequential(*self.layers)

        self.last_layer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x):
        x = self.first_layer(x)

        x_out = self.layers[0](x)
        for layer in self.layers[1:]:
                x_out = x_out + layer(x)

        x = self.last_layer(x_out)
        return x

def test():
    x = torch.rand(2, 128, 240)
    convnet = ConvNet(
        in_channels=128,
        out_channels=128,
        hidden_size=128,
        n_stacks=2,
        n_blocks=2,
        kernel_size=3,
        activation="ReLU",
    )
    y = convnet(x)
    print(y.shape)

    convnet = ConvNetDouble(
        in_channels=128,
        out_channels=128,
        hidden_size=128,
        n_layers=2,
        n_stacks=2,
        n_blocks=2,
        middle_layer=nn.MaxPool1d(
            kernel_size=8,
            stride=8,
        ),
        kernel_size=3,
        activation="ReLU",
    )
    y = convnet(x)
    print(y.shape)

