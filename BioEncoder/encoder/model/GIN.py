from dgl.nn.pytorch import WeightAndSum
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling
class DGL_GIN(nn.Module):
    def __init__(self, node_feat_dim=20, edge_feat_dim=1, output_feats=64, device='cpu', max_nodes=50, readout = True):
        super(DGL_GIN, self).__init__()
        self.device = device
        self.gnn = GIN(node_feat_dim = node_feat_dim,
                       edge_feat_dim=edge_feat_dim
                       )
        self.readout = AvgPooling()
        self.transform = nn.Linear(300, output_feats)
        self.output_dim = 64
        # gnn_out_feats = self.gnn.hidden_feats[-1]
        # self.readout = readout
        # self.readout_1D = WeightedSumAndMaxTransform(gnn_out_feats, output_feats)
        # self.transform = nn.Linear(gnn_out_feats * 2, output_feats)
        # self.output_dim = output_feats
        # self.max_nodes = max_nodes
        # self.virtual_node_feat = torch.zeros(gnn_out_feats).to(self.device)  # Assuming it should be zeros

    def forward(self, bg):


        bg = bg.to(self.device)
        feats = bg.ndata.pop('h')
        e_feats = bg.edata.pop('e')

        feats = feats.to(torch.float32)
        e_feats = e_feats.to(torch.float32)

        node_feats = self.gnn(feats, e_feats)
        graph_feats = self.readout(bg, node_feats)
        return self.transform(graph_feats)

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


class GINLayer(nn.Module):
    # Removing num_edge_emb_list since we no longer need edge embeddings
    def __init__(self, emb_dim, batch_norm=True, activation=None):
        super(GINLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )

        if batch_norm:
            self.bn = nn.BatchNorm1d(emb_dim)
        else:
            self.bn = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats, continuous_edge_feats):
        g = g.local_var()
        g.ndata['feat'] = node_feats
        g.edata['feat'] = continuous_edge_feats  # Directly use continuous edge features
        g.update_all(fn.u_add_e('feat', 'feat', 'm'), fn.sum('m', 'feat'))

        node_feats = self.mlp(g.ndata.pop('feat'))
        if self.bn is not None:
            node_feats = self.bn(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats

#... [Rest of the code remains mostly unchanged]

class GIN(nn.Module):
    # Update edge_feat_dim to be continuous dimension size
    def __init__(self, node_feat_dim, edge_feat_dim, num_layers=5, emb_dim=300, JK='last', dropout=0.5):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.JK = JK
        self.dropout = nn.Dropout(dropout)

        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.gnn_layers.append(GINLayer(node_feat_dim, emb_dim))
            else:
                self.gnn_layers.append(GINLayer(emb_dim, emb_dim))

# Please note that I've assumed the GINLayer is defined elsewhere in your code as you haven't shared its definition.
