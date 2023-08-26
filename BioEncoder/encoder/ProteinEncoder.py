from BioEncoder.encoder.config import *
from BioEncoder.util.data.data_processing import *
import copy


class ProteinEncoder:
    def __init__(self, method, **config):
        self.featurizer, self.model, self.model_training_setup = init_protein_method(method, config)

    def get_model(self):
        return self.model

    def get_featurizer(self):
        return self.featurizer

    def transform(self, proteins, idx=None):
        temp = copy.deepcopy(proteins)
        return apply_transform(temp, self.featurizer, idx)

    def get_num_transform(self):
        return self.featurizer.get_num_steps()

    def get_output_dim(self):
        return self.model.output_dim




