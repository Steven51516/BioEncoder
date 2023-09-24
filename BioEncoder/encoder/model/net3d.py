from typing import List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn



class Net3D(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim = 74, target_dim = 128, readout_aggregators: List[str] = ["min", "max", "mean"], batch_norm=True,
                 node_wise_output_layers=0, readout_batchnorm=True, batch_norm_momentum=0.93, reduce_func='mean',
                 dropout=0.1, propagation_depth: int = 2, readout_layers: int = 1, readout_hidden_dim=40,
                 fourier_encodings=4, activation: str = 'SiLU', update_net_layers=2, message_net_layers=1, use_node_features=True, **kwargs):
        super(Net3D, self).__init__()
        self.fourier_encodings = fourier_encodings
        edge_in_dim = 1 if fourier_encodings == 0 else 2 * fourier_encodings + 1
        self.edge_input = MLP(in_dim=edge_in_dim, hidden_size=hidden_dim, out_dim=hidden_dim, mid_batch_norm=batch_norm,
                              last_batch_norm=batch_norm, batch_norm_momentum=batch_norm_momentum, layers=1,
                              mid_activation=activation, dropout=dropout, last_activation=activation,
                              )

        self.use_node_features = use_node_features
        if self.use_node_features:
            self.atom_encoder = AtomEncoder(hidden_dim)
        else:
            self.node_embedding = nn.Parameter(torch.empty((hidden_dim,)))
            nn.init.normal_(self.node_embedding)

        self.mp_layers = nn.ModuleList()
        self.output_dim = 128
        for _ in range(propagation_depth):
            self.mp_layers.append(
                Net3DLayer(edge_dim=hidden_dim, hidden_dim=hidden_dim, batch_norm=batch_norm,
                           batch_norm_momentum=batch_norm_momentum, dropout=dropout, mid_activation=activation,
                           reduce_func=reduce_func, message_net_layers=message_net_layers,
                           update_net_layers=update_net_layers))

        self.node_wise_output_layers = node_wise_output_layers
        if self.node_wise_output_layers > 0:
            self.node_wise_output_network = MLP(in_dim=hidden_dim, hidden_size=hidden_dim, out_dim=hidden_dim,
                                                mid_batch_norm=batch_norm, last_batch_norm=batch_norm,
                                                batch_norm_momentum=batch_norm_momentum, layers=node_wise_output_layers,
                                                mid_activation=activation, dropout=dropout, last_activation='None',
                                                )

        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, batch_norm_momentum=batch_norm_momentum, out_dim=target_dim,
                          layers=readout_layers)
        self.device = "cpu"

    def forward(self, graph: dgl.DGLGraph):
        graph = graph.to(self.device)
        if self.use_node_features:
            graph.ndata['feat'] = graph.ndata['h'] #self.atom_encoder(graph.ndata['h'])
        else:
            graph.ndata['feat'] = self.node_embedding[None, :].expand(graph.number_of_nodes(), -1)

        if self.fourier_encodings > 0:
            graph.edata['d'] = fourier_encode_dist(graph.edata['d'], num_encodings=self.fourier_encodings)
        graph.apply_edges(self.input_edge_func)

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        if self.node_wise_output_layers > 0:
            graph.apply_nodes(self.output_node_func)

        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)

    def output_node_func(self, nodes):
        return {'feat': self.node_wise_output_network(nodes.data['feat'])}

    def input_edge_func(self, edges):
        return {'d': F.silu(self.edge_input(edges.data['d']))}


class Net3DLayer(nn.Module):
    def __init__(self, edge_dim, reduce_func, hidden_dim, batch_norm, batch_norm_momentum, dropout,
                 mid_activation, message_net_layers, update_net_layers):
        super(Net3DLayer, self).__init__()
        self.message_network = MLP(in_dim=hidden_dim * 2 + edge_dim, hidden_size=hidden_dim, out_dim=hidden_dim,
                                   mid_batch_norm=batch_norm, last_batch_norm=batch_norm,
                                   batch_norm_momentum=batch_norm_momentum, layers=message_net_layers,
                                   mid_activation=mid_activation, dropout=dropout, last_activation=mid_activation,
                                   )
        if reduce_func == 'sum':
            self.reduce_func = fn.sum
        elif reduce_func == 'mean':
            self.reduce_func = fn.mean
        else:
            raise ValueError('reduce function not supported: ', reduce_func)

        self.update_network = MLP(in_dim=hidden_dim, hidden_size=hidden_dim, out_dim=hidden_dim,
                                  mid_batch_norm=batch_norm, last_batch_norm=batch_norm,
                                  batch_norm_momentum=batch_norm_momentum, layers=update_net_layers,
                                  mid_activation=mid_activation, dropout=dropout, last_activation='None',
                                  )

        self.soft_edge_network = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        graph.update_all(message_func=self.message_function, reduce_func=self.reduce_func(msg='m', out='m_sum'),
                         apply_node_func=self.update_function)

    def message_function(self, edges):
        message_input = torch.cat(
            [edges.src['feat'], edges.dst['feat'], edges.data['d']], dim=-1)
        message = self.message_network(message_input)
        edges.data['d'] += message
        edge_weight = torch.sigmoid(self.soft_edge_network(message))
        return {'m': message * edge_weight}

    def update_function(self, nodes):
        h = nodes.data['feat']
        input = torch.cat([nodes.data['m_sum'] + nodes.data['feat']], dim=-1)
        h_new = self.update_network(input)
        output = h_new + h
        return {'feat': output}


full_atom_feature_dims = list(range(0, 73))
class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for i, dim in enumerate(full_atom_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def reset_parameters(self):
        for i, embedder in enumerate(self.atom_embedding_list):
            embedder.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            if self.padding:
                x_embedding += self.atom_embedding_list[i](x[:, i].long() + 1)
            else:
                x_embedding += self.atom_embedding_list[i](x[:, i].long())

        return x_embedding


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x.squeeze()


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:
    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)
    Arguments
    ----------
        in_dim: int
            Input dimension of the layer (the torch.nn.Linear)
        out_dim: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        batch_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_dim}}`
            (Default value = None)
    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        batch_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_dim: int
            Input dimension of the linear layer
        out_dim: int
            Output dimension of the linear layer
    """

    def __init__(self, in_dim, out_dim, activation='relu', dropout=0., batch_norm=False, batch_norm_momentum=0.1,
                 bias=True, init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()
        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.linear = nn.Linear(in_dim, out_dim, bias=bias).to(device)
        self.dropout = None
        self.batch_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_dim, momentum=batch_norm_momentum).to(device)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_dim)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.batch_norm is not None:
            if h.shape[1] != self.out_dim:
                h = self.batch_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.batch_norm(h)
        return h


class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(self, in_dim, out_dim, layers, hidden_size=None, mid_activation='relu', last_activation='none',
                 dropout=0., mid_batch_norm=False, last_batch_norm=False, batch_norm_momentum=0.1, device='cpu'):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(in_dim, out_dim, activation=last_activation, batch_norm=last_batch_norm,
                                                device=device, dropout=dropout,
                                                batch_norm_momentum=batch_norm_momentum))
        else:
            self.fully_connected.append(
                FCLayer(in_dim, hidden_size, activation=mid_activation, batch_norm=mid_batch_norm,
                        device=device, dropout=dropout, batch_norm_momentum=batch_norm_momentum))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_size, hidden_size, activation=mid_activation,
                                                    batch_norm=mid_batch_norm, device=device, dropout=dropout,
                                                    batch_norm_momentum=batch_norm_momentum))
            self.fully_connected.append(
                FCLayer(hidden_size, out_dim, activation=last_activation, batch_norm=last_batch_norm,
                        device=device, dropout=dropout, batch_norm_momentum=batch_norm_momentum))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x


SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'SiLU', 'None'}
EPS = 1e-5


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()