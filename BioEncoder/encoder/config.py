from .model import *
from .featurizer import *


def init_drug_method(method, user_config):
    model_training_setup = {"collate_func": None, "to_device_in_model": False}
    if method == "CNN":
        config = {"in_channel": 63, "input_length": DrugOneHotFeaturizer.MAX_SEQ_DRUG}
        config.update(user_config)
        return DrugOneHotFeaturizer(), CNN(**config), model_training_setup
    elif method == "GCN":
        from BioEncoder.util.data.collate_func import dgl_collate_func
        config = user_config
        model_training_setup['collate_func'] = dgl_collate_func
        model_training_setup['to_device_in_model'] = True
        return GCNGraphFeaturizer(), DGL_GCN(**config), model_training_setup
    elif method == "Transformer":
        config = {
            'emb_size': 128,
            'intermediate_size': 512,
            'num_attention_heads': 8,
            'n_layer': 8,
            'dropout_rate': 0.1,
            'attention_probs_dropout': 0.1,
            'hidden_dropout_rate': 0.1,
            'emb_max_pos_size': 50,
            'input_dim': 8420
        }
        config.update(user_config)
        model_training_setup['to_device_in_model'] = True
        return DrugEmbeddingFeaturizer(), Transformer(**config), model_training_setup
    elif method == "Morgan":
        config = user_config
        return MorganFeaturizer(), MLP(**config), model_training_setup

    else:
        return None, None, None


def init_protein_method(method, user_config):
    model_training_setup = {"collate_func": None, "to_device_in_model": False}
    if method == "CNN":
        config = {"in_channel": 26, "input_length": ProteinOneHotFeaturizer.MAX_SEQ_PROTEIN}
        config.update(user_config)
        return ProteinOneHotFeaturizer(), CNN(**config), model_training_setup
    elif method == "Transformer":
        config = {
            'emb_size': 64,
            'intermediate_size': 256,
            'num_attention_heads': 4,
            'n_layer': 2,
            'dropout_rate': 0.1,
            'attention_probs_dropout': 0.1,
            'hidden_dropout_rate': 0.1,
            'emb_max_pos_size': 545,
            'input_dim': 8420
        }
        config.update(user_config)
        model_training_setup['to_device_in_model'] = True
        return ProteinEmbeddingFeaturizer(), Transformer(**config), model_training_setup
    else:
        return None, None, None
