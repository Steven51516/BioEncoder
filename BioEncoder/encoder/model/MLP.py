import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    def __init__(self, input_dim=1024, output_dim=256, hidden_dims_lst = [1024, 256, 64]):
        '''
            input_dim (int)
            output_dim (int)
            hidden_dims_lst (list, each element is a integer, indicating the hidden size)

        '''
        super(MLP, self).__init__()
        layer_size = len(hidden_dims_lst) + 1
        dims = [input_dim] + hidden_dims_lst + [output_dim]
        self.output_shape = output_dim
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

    def forward(self, v):
        # predict
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v