import torch.nn as nn
import torch
import torch.nn.functional as F



class GraphConv(nn.Module):
    """Graph convolution layer.
    Xnew = activation(AXW + b)
        Args:
            d_input: int, the input dimension (number of features per node).
            d_model: int, the output dimension.
            use_bias: bool, whether the bias is used.
            activation: str or callable, the activation function.

        Inputs:
            a: Adjacency matrix A. shape = `(batch_size, n, n)`
            x: Input matrix X. shape = `(batch_size, n, d_input)`

        Outputs:
            xnew: Updated feature matrix X_{i+1}. shape = `(batch_size, n, d_model)`
    """

    def __init__(self, d_input, d_model, use_bias=True, activation=None):
        super(GraphConv, self).__init__()
        self.d_model = d_model
        self.use_bias = use_bias
        self.activation = activation

        # Define the dense (fully connected) layer
        self.dense = nn.Linear(d_input, d_model, bias=use_bias)


        # Define the activation function
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        # Add more activations if needed
        # elif activation == 'sigmoid':
        #     self.activation_fn = nn.Sigmoid()
        else:
            self.activation_fn = None

    def forward(self, a, x):
        ax = torch.matmul(a, x)
        batch_size, n, d_input = ax.shape

        # Reshape only the last two dimensions for the Linear layer
        ax_reshaped = ax.view(-1, d_input)
        z = self.dense(ax_reshaped)

        # Reshape back to original shape
        z = z.view(batch_size, n, -1)

        # Apply the activation function if it exists
        if self.activation_fn:
            z = self.activation_fn(z)


        return z


class Drug3DEncoder(nn.Module):
    def __init__(self, d_model = 128, n_layers_2d=2, n_layers_3d=1):
        super(Drug3DEncoder, self).__init__()
        # First GCN layer for 2D with d_input as 34
        self.d_model = d_model
        self.gcn_2d = [GraphConv(34, d_model, activation='relu')]
        # Subsequent GCN layers for 2D
        self.gcn_2d.extend([GraphConv(d_model, d_model, activation='relu') for _ in range(n_layers_2d - 1)])
        self.gcn_2d = nn.ModuleList(self.gcn_2d)

        # First GCN layer for 3D with d_input as 34
        self.gcn_3d = [GraphConv(34, d_model, activation='relu')]
        # Subsequent GCN layers for 3D
        self.gcn_3d.extend([GraphConv(d_model, d_model, activation='relu') for _ in range(n_layers_3d - 1)])
        self.gcn_3d = nn.ModuleList(self.gcn_3d)
        self.fc = nn.Linear(2 * d_model, d_model)
        self.output_shape = 128

        self.attention = nn.Linear(d_model, 1)


    def forward(self, xx):
        a = xx[1]
        s = xx[2]
        x = xx[0]
        #
        # a = torch.tensor(a)
        # s = torch.tensor(s)
        # x = torch.tensor(x)


        ha = x
        for i in range(len(self.gcn_2d)):
            ha = self.gcn_2d[i](a, ha)

        hs = x
        for i in range(len(self.gcn_3d)):
            hs = self.gcn_3d[i](s, hs)

        h = torch.cat([ha, hs], dim=-1)
        h_reshaped = h.view(-1, self.d_model * 2)
        z = self.fc(h_reshaped)
        z = z.view(h.size(0), h.size(1), -1)


        # new added
        attention_scores = self.attention(z)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
        z_weighted_sum = torch.sum(z * attention_weights, dim=1)

        z_max, _ = torch.max(z, dim=1)

        # Combine the weighted sum and max outputs
        z_final = z_weighted_sum + z_max  # Shape will be (batch_size, d_model)

        return z_max


class Drug2DEncoder(nn.Module):
    def __init__(self, d_input, d_model):
        super(Drug2DEncoder, self).__init__()
        self.gcn = GraphConv(d_input, d_model)

    def forward(self, a, x):
        h = self.gcn(a, x)
        return h
