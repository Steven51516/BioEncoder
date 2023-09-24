from .model import *
from .featurizer import *


def init_drug_method(method, user_config):
    model_training_setup = {"collate_func": None, "to_device_in_model": False}
    if method == "CNN":
        model_training_setup["loadtime_transform"] = True
        config = {"in_channel": 63, "input_length": DrugOneHotFeaturizer.MAX_SEQ_DRUG}
        config.update(user_config)
        return DrugOneHotFeaturizer(), CNN(**config), model_training_setup
    elif method == "GCN":
        import dgl
        config = user_config
        model_training_setup['collate_func'] = dgl.batch
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
        from BioEncoder.util.data.collate_func import tuple_collate
        model_training_setup['collate_func'] = tuple_collate
        return DrugEmbeddingFeaturizer(), Transformer(**config), model_training_setup
    elif method == "Morgan":
        config = user_config
        return MorganFeaturizer(), MLP(**config), model_training_setup
    elif method == "3D":
        from .model.Drug3D import Drug3DEncoder
        from BioEncoder.util.data.collate_func import tuple_collate
        model_training_setup['to_device_in_model'] = True
        model_training_setup['collate_func'] = tuple_collate
        config = user_config
        return Drug3DFeaturizer(), Drug3DEncoder(**config), model_training_setup
    elif method == "Schnet":
        from .model.Schnet import SchNet
        from torch_geometric.data import Batch
        config = user_config
        model_training_setup["collate_func"] = Batch.from_data_list
        model_training_setup["to_device_in_model"] = True
        return DrugMolNetFeaturizer(), SchNet(**config), model_training_setup
    elif method == "3dNet":
        from .model.net3d import Net3D
        from BioEncoder.util.data.collate_func import dgl_collate_func
        model_training_setup['collate_func'] = dgl_collate_func
        model_training_setup['to_device_in_model'] = True
        config = user_config
        return Drug3dNetFeaturizer(), Net3D(0,0,**config), model_training_setup
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
        from BioEncoder.util.data.collate_func import tuple_collate
        model_training_setup['collate_func'] = tuple_collate
        return ProteinEmbeddingFeaturizer(), Transformer(**config), model_training_setup
    elif method == "AAD":
        config = {
            "input_dim": 8420
        }
        config.update(user_config)
        return ProteinAADFeaturizer(), MLP(**config), model_training_setup
    else:
        return None, None, None



def init_protein_pdb_method(method, user_config):
    model_training_setup = {"collate_func": None, "to_device_in_model": False}
    if method == "GCN":
        config = {
            "in_feats" : 20,
            "output_feats" : 64
        }
        config.update(user_config)
        import dgl
        model_training_setup['collate_func'] = dgl.batch
        model_training_setup['to_device_in_model'] = True
        return ProteinGCNFeaturizer(), DGL_GCN(**config), model_training_setup
    elif method == "GCN_ESM":
        config = {
            "in_feats": 1280,
            "output_feats": 128
        }
        config.update(user_config)
        import dgl
        model_training_setup['collate_func'] = dgl.batch
        model_training_setup['to_device_in_model'] = True
        return ProteinGCN_ESMFeaturizer(), DGL_GCN(**config), model_training_setup
    elif method == "ESM":
        config = {
            "input_dim": 1280
        }
        config.update(user_config)
        return ProteinESMFeaturizer(), MLP(**config), model_training_setup
    else:
        return None, None, None


