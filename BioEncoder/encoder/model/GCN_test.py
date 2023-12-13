from dgl.nn.pytorch import WeightAndSum
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F

class DGL_GCN_Test(nn.Module):
    def __init__(self, in_feats=74, hidden_feats=[64,64,64], activation=[F.relu,F.relu,F.relu], output_feats=64, device='cpu', max_nodes=50, readout = True):
        super(DGL_GCN_Test, self).__init__()
        self.device = device
        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       activation=activation
                       )
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = readout
        self.readout_1D = WeightedSumAndMaxTransform(gnn_out_feats, output_feats)
        self.transform = nn.Linear(gnn_out_feats * 2, output_feats)
        self.output_shape = output_feats
        self.max_nodes = max_nodes
        self.virtual_node_feat = torch.zeros(gnn_out_feats).to(self.device)  # Assuming it should be zeros
        self.dim_reduction_layer = nn.Linear(384, 54)

    def forward(self, bg):


        bg = bg.to(self.device)
        feats = bg.ndata.pop('bert')
        feats2 = bg.ndata.pop('h')
        feats = feats.to(torch.float32)
        feats = self.dim_reduction_layer(feats)
        feats2 = feats2.to(torch.float32)
        feats = torch.cat((feats, feats2), dim=1)

        node_feats = self.gnn(bg, feats)
        if self.readout:
            # batch_size = bg.batch_size
            # return node_feats.view(batch_size, -1, self.output_dim)
            return self.readout_1D(bg, node_feats)
        else:
            return bg, node_feats

    def readout_2D(self, bg, node_feats):

        num_nodes_per_graph = bg.batch_num_nodes()

        # Determine the maximum number of nodes across all graphs
        max_nodes = max(num_nodes_per_graph)

        # Number of graphs in the batch
        num_graphs = len(num_nodes_per_graph)

        # Node feature dimension
        node_feat_dim = node_feats.shape[-1]

        # Initialize output tensor with zeros
        output = torch.zeros((num_graphs, max_nodes, node_feat_dim), device=node_feats.device)

        # Counter for where in the node_feats tensor we currently are
        node_counter = 0
        for i, num_nodes in enumerate(num_nodes_per_graph):
            # Get node features for the current graph
            graph_node_feats = node_feats[node_counter:node_counter + num_nodes]

            # Set them in the output tensor
            output[i, :num_nodes] = graph_node_feats

            # Update the counter
            node_counter += num_nodes

        return output


class WeightedSumAndMaxTransform(nn.Module):
    def __init__(self, gnn_out_feats, output_feats):
        super(WeightedSumAndMaxTransform, self).__init__()
        self.weighted_sum_and_max = WeightedSumAndMax(gnn_out_feats)
        self.linear = nn.Linear(gnn_out_feats * 2, output_feats)

    def forward(self, bg, feat):
        x = self.weighted_sum_and_max(bg, feat)
        return self.linear(x)



# pylint: disable=W0221
class WeightedSumAndMax(nn.Module):
    r"""Apply weighted sum and max pooling to the node
    representations and concatenate the results.

    Parameters
    ----------
    in_feats : int
        Input node feature size
    """

    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):
        """Readout

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        h_g : FloatTensor of shape (B, 2 * M1)
            * B is the number of graphs in the batch
            * M1 is the input node feature size, which must match
              in_feats in initialization
        """
        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g


import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


# pylint: disable=W0221, C0103
class GCNLayer(nn.Module):
    r"""Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    out_feats : int
        Number of output node features.
    gnn_norm : str
        The message passing normalizer, which can be `'right'`, `'both'` or `'none'`. The
        `'right'` normalizer divides the aggregated messages by each node's in-degree.
        The `'both'` normalizer corresponds to the symmetric adjacency normalization in
        the original GCN paper. The `'none'` normalizer simply sums the messages.
        Default to be 'none'.
    activation : activation function
        Default to be None.
    residual : bool
        Whether to use residual connection, default to be True.
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph. Defaults to False.
    """

    def __init__(
            self,
            in_feats,
            out_feats,
            gnn_norm="none",
            activation=None,
            residual=True,
            batchnorm=True,
            dropout=0.0,
            allow_zero_in_degree=False,
    ):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(
            in_feats=in_feats,
            out_feats=out_feats,
            norm=gnn_norm,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match in_feats in initialization

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output node feature size, which must match out_feats in initialization
        """
        new_feats = self.graph_conv(g, feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class GCN(nn.Module):
    r"""GCN from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers.  By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If not None, ``activation[i]`` gives the activation function to be used for
        the i-th GCN layer. ``len(activation)`` equals the number of GCN layers.
        By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph for all layers. By default, will not
        allow zero in degree nodes.
    """

    def __init__(
            self,
            in_feats,
            hidden_feats=None,
            gnn_norm=None,
            activation=None,
            residual=None,
            batchnorm=None,
            dropout=None,
            allow_zero_in_degree=None,
    ):
        super(GCN, self).__init__()

        if hidden_feats is None:
            hidden_feats = [64, 64]

        n_layers = len(hidden_feats)
        if gnn_norm is None:
            gnn_norm = ["none" for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0.0 for _ in range(n_layers)]
        lengths = [
            len(hidden_feats),
            len(gnn_norm),
            len(activation),
            len(residual),
            len(batchnorm),
            len(dropout),
        ]
        assert len(set(lengths)) == 1, (
            "Expect the lengths of hidden_feats, gnn_norm, "
            "activation, residual, batchnorm and dropout to "
            "be the same, got {}".format(lengths)
        )

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                GCNLayer(
                    in_feats,
                    hidden_feats[i],
                    gnn_norm[i],
                    activation[i],
                    residual[i],
                    batchnorm[i],
                    dropout[i],
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] in initialization.
        """
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats
