import torch.nn as nn
import torch.nn.functional as F
import torch


class CausalConv1D(nn.Module):

    """
    Causal 1D Convolution Layer with dilation and padding to ensure causality.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, dilation=dilation)

    def forward(self, x):
        # Pad the input on the left to ensure causality
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)
    

class WaveNetBlock(nn.Module):

    """
    A single block of the WaveNet architecture, consisting of a causal convolution,
    followed by a gated activation unit and a residual connection.
    """
    def __init__(self, channels, kernel_size, dilation):
        super(WaveNetBlock, self).__init__()
        self.causal_conv = CausalConv1D(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1D(channels, channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # Gated activation
        gate = torch.tanh(self.causal_conv(x))
        filter = torch.sigmoid(self.gate_conv(x))
        out = gate * filter  # Element-wise multiplication
        skip = self.skip(out)
        residual = self.residual_conv(out) + x[:, :, -out.size(-1):]
        return residual, skip  # Residual connection and skip connection
    

class CausalWaveNet(nn.Module):

    def __init__(self, in_channels=4, hidden_channels=64, kernel_size=3,
                 dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], out_channels=2):
        super(CausalWaveNet, self).__init__()
        self.input_conv = CausalConv1D(in_channels, hidden_channels, kernel_size, dilation=1)

        self.blocks = nn.ModuleList()

        for d in dilations:
             self.blocks.append(WaveNetBlock(hidden_channels, kernel_size, d))

        # Output layer: predict the last time step's clean values
        self.output_conv = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

        # Cousal layer to aggregate skip connections
        self.final = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # X: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.input_conv(x)

        skip_sun = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sun += skip

        out = self.final(skip_sun)[:, :, -1]  # Take the last time step's output
        return out
    

    

