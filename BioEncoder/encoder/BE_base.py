from .BE_config import *
import torch.nn as nn


class BENode:

    def __init__(self, parents=[], children=[], model=None, root_idx=None,
                 output_shape=None):
        super(BENode, self).__init__()
        self.parents = parents
        self.children = children
        self.model = model
        self.output_shape = output_shape
        self.root_idx = root_idx
        for parent in parents:
            parent.add_child(self)

    def add_parent(self, node):
        if node not in self.parents:
            self.parents.append(node)

    def add_child(self, node):
        if node not in self.children:
            self.children.append(node)

    def get_children(self):
        return self.children

    def get_parents(self):
        return self.parents

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return len(self.parents) == 0

    def get_model(self):
        return self.model

    def get_output_shape(self):
        return self.output_shape

    def get_root_idx(self):
        return self.root_idx


class Encoder:
    def __init__(self, method, **config):
        self.method = method
        self.featurizer = None
        self.model = None
        self.model_training_setup = None
        self.output_shape = None

    def get_model(self):
        return self.model

    def get_featurizer(self):
        return self.featurizer

    def transform(self, entities, mode="default"):
        temp = copy.deepcopy(entities)
        return self.featurizer(temp, mode)

    def get_output_shape(self):
        return self.model.output_shape


class Interaction(nn.Module):
    def __init__(self, nodes, method, head=None, mlp_hidden_layers=None, **config):
        super().__init__()
        self.method = method
        p_output_shapes = [node.get_output_shape() for node in nodes]
        self.inter_layer, self.output_shape = init_inter_layer(method, p_output_shapes, **config)
        self.mlp = self.setup_mlp(head, mlp_hidden_layers) if head else None

    def setup_mlp(self, head, mlp_hidden_layers):
        input_dim = self.output_shape
        if mlp_hidden_layers is None:
            mlp_hidden_layers = [1024, 718, 512]
        mlp_layers = []
        for hidden_dim in mlp_hidden_layers:
            mlp_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        mlp_layers.append(nn.Linear(input_dim, head))
        return nn.Sequential(*mlp_layers)

    def get_output_shape(self):
        return self.output_shape

    def forward(self, *encoded):
        out = torch.cat(encoded, dim=1) if self.method == "cat" else self.inter_layer(*encoded)
        return self.mlp(out) if self.mlp else out
