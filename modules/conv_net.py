
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(
            self, 
            in_channels: int = 128,
            out_channels: int = 512,
            kernel_size: int = 5,
            layers: int = 5,
            activation: str = 'GELU',
            type: str = 'encoder'
    ):
        super(ConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.kernel_size = kernel_size
        self.activation = activation

        self.blocks = nn.ModuleList()

        chs = self.in_channels
        if type == 'encoder':
            self.add_down_up_sampling()
            chs = self.out_channels
        
        for i in range(self.layers - 1):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=chs,
                    out_channels=chs,
                    kernel_size=self.kernel_size,
                    padding=2
                ),
                getattr(nn, self.activation)(),
            ))
        if type == 'decoder':
            self.add_down_up_sampling()

    def add_down_up_sampling(self):
        self.blocks.append(nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=2
            ),
            nn.BatchNorm1d(self.out_channels),
            getattr(nn, self.activation)(),
        ))


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ConvNetEncoder(ConvNet):
    def __init__(
            self, 
            in_channels: int = 128,
            out_channels: int = 512,
            kernel_size: int = 5,
            layers: int = 5,
            activation: str = 'GELU',
    ):
        super(ConvNetEncoder, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            layers=layers,
            activation=activation,
            type='encoder'
        )

class ConvNetDecoder(ConvNet):
    def __init__(
            self, 
            in_channels: int = 512,
            out_channels: int = 128,
            kernel_size: int = 5,
            layers: int = 5,
            activation: str = 'GELU',
    ):
        super(ConvNetDecoder, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            layers=layers,
            activation=activation,
            type='decoder'
        )
    


if __name__ == '__main__':

    B = 3
    T = 256

    encoder = ConvNetEncoder()

    print(encoder)

    x = torch.randn(B, 128, T)
    y = encoder(x)

    print(y.shape)

    decoder = ConvNetDecoder()
    print(decoder)

    x = torch.randn(B, 512, T)
    y = decoder(x)

    print(y.shape)
