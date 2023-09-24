import copy
from .config import *
import torch.nn as nn

from ..util.data.data_processing import apply_transform


class BENode:

    def __init__(self, parents=[], children=[], model=None, output_layer=None, input_type=None,
                 output_dim=128):
        super(BENode, self).__init__()
        self.parents = parents
        self.children = children
        self.model = model
        self.output_layer = output_layer
        self.input_type = input_type
        self.output_dim = output_dim

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

    def get_output_dim(self):
        return self.output_dim

    def set_output_dim(self, dim):
        self.output_dim = dim

    def get_input_type(self):
        return self.input_type


class Encoder:
    def __init__(self, method, **config):
        self.method = method
        self.config = config
        self.featurizer = None
        self.model = None
        self.model_training_setup = None

    def get_model(self):
        return self.model

    def get_featurizer(self):
        return self.featurizer

    def transform(self, entities, idx=None):
        temp = copy.deepcopy(entities)
        return apply_transform(temp, self.featurizer, idx)

    def get_num_transform(self):
        if self.featurizer:
            return self.featurizer.get_num_steps()
        raise ValueError("Featurizer not initialized.")

    def get_output_dim(self):
        return self.model.output_dim


class Interaction(nn.Module):
    def __init__(self, node1, node2, method, head=None, mlp_hidden_layers=None, **config):
        nn.Module.__init__(self)
        self.method = method
        if self.method == "cat":
            self.join_dim = node1.get_output_dim() + node2.get_output_dim()
            self.output_dim = self.join_dim
        if self.method == "bilinear":
            from BioEncoder.encoder.Interaction.bilinear import BANLayer
            from torch.nn.utils.weight_norm import weight_norm
            self.bilinear = weight_norm(
            BANLayer(v_dim=64, q_dim=64, h_dim=256, h_out=2),
            name='h_mat', dim=None)
            self.join_dim =256
            self.output_dim = 256
        elif self.method == "fusion":
            from BioEncoder.encoder.Interaction.bilinear_fusion import BilinearFusion
            a = node1.get_output_dim()
            b =  node2.get_output_dim()
            self.bilinear_fusion = BilinearFusion(dim1=a, dim2 = b, mmhid1=a*b, mmhid2=128)
            self.output_dim = 128
            self.join_dim = 128

        if head:
            if mlp_hidden_layers is None:
                mlp_hidden_layers = [1024, 718, 512]
            mlp_layers = []
            input_dim = self.join_dim
            for hidden_dim in mlp_hidden_layers:
                mlp_layers.append(nn.Linear(input_dim, hidden_dim))
                mlp_layers.append(nn.ReLU())  # activation function
                input_dim = hidden_dim
            if head:
                mlp_layers.append(nn.Linear(input_dim, head))
            self.mlp = nn.Sequential(*mlp_layers)
        else:
            self.mlp = None

    def forward(self, encoded1, encoded2):
        if self.method == "cat":
            out = torch.cat([encoded1, encoded2], dim=1)
        elif self.method == "bilinear":
            out = self.bilinear(encoded1, encoded2)
        elif self.method == "fusion":
            out = self.bilinear_fusion(encoded1, encoded2)
        if self.mlp:
            out = self.mlp(out)
        return out
