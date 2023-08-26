import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channel, input_length, filters=[32, 64, 96], output_feats=256, kernels=[4, 6, 8]):
        super(CNN, self).__init__()
        input_size = (in_channel, input_length)
        channels = [in_channel] + filters
        self.conv = nn.ModuleList([
            nn.Conv1d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernels[i])
            for i in range(len(filters))
        ])
        n_size = self._conv_output_size(input_size)
        self.fc = nn.Linear(n_size, output_feats)
        self.output_dim = output_feats

    def _conv_output_size(self, shape):
        input_tensor = torch.rand(1, *shape)
        output_feat = self._forward_conv(input_tensor)
        n_size = output_feat.view(1, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        for conv in self.conv:
            x = F.relu(conv(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
