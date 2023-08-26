from .config import *
from BioEncoder.util.data.data_processing import *
from .Interaction import CombinedEncoder
import copy


class DrugEncoder:
    def __init__(self, method, **config):
        self.featurizer, self.model, self.model_training_setup = init_drug_method(method, config)

    def get_model(self):
        return self.model

    def get_featurizer(self):
        return self.featurizer

    def transform(self, drugs, idx=None):
        temp = copy.deepcopy(drugs)
        return apply_transform(temp, self.featurizer, idx)

    def get_num_transform(self):
        return self.featurizer.get_num_steps()

    def get_joined_model(self, encoder, join_method="cat", head=None, mlp_hidden_layers=None):
        return CombinedEncoder(self, encoder, join_method, head, mlp_hidden_layers)

    def get_output_dim(self):
        return self.model.output_dim

