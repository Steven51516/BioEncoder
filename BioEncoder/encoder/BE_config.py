from .model import *
from .featurizer import *
import inspect
import dgl

#DELETE
from .model.Drug3D import Drug3DEncoder
from .model.Pocket import DTITAG
from BioEncoder.util.data.collate_func import tuple_collate

from BioEncoder.encoder.Interaction.attention import MultiHeadAttentionInteract
from BioEncoder.encoder.Interaction.bilinear import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from BioEncoder.encoder.Interaction.bilinear_fusion import BilinearFusion

def process_configs(model_class, config):
    """
    Splits the configuration parameters between the model and featurizer.

    Args:
    - model_class: The class of the model.
    - config: The combined configuration dictionary.

    Returns:
    - Tuple of two dictionaries: (model_params, featurizer_params).
    """
    model_args = inspect.signature(model_class.__init__).parameters
    model_params = {k: v for k, v in config.items() if k in model_args}
    featurizer_params = {k: v for k, v in config.items() if k not in model_args}
    return model_params, featurizer_params


def init_drug_method(method, user_config):
    model_class, featurizer_class = {
        "CNN": (CNN, DrugOneHotFeaturizer),
        "GCN": (DGL_GCN_Test, DrugChemBertGNNFeaturizer),  # Update with correct class
        "Transformer": (Transformer, DrugEmbeddingFeaturizer),
        "Morgan": (MLP, MorganFeaturizer),
        "3D": (Drug3DEncoder, Drug3DFeaturizer),
    }.get(method, (None, None))

    if model_class is None:
        raise ValueError(f"Unsupported method: {method}")

    default_model_config = {}
    default_featurizer_config = {}
    model_training_setup = {"collate_func": None, "to_device_in_model": False, "loadtime_transform": False}

    if method == "CNN":
        default_model_config = {"in_channel": 63, "input_length": DrugOneHotFeaturizer.MAX_SEQ_DRUG}
        model_training_setup["loadtime_transform"] = True
    elif method == "GCN":
        default_model_config = {"in_feats": 74+54}
        model_training_setup['collate_func'] = dgl.batch
        model_training_setup['to_device_in_model'] = True
    elif method == "Transformer":
        default_model_config = {
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
        model_training_setup['to_device_in_model'] = True
        model_training_setup['collate_func'] = tuple_collate
    elif method == "Morgan":
        pass
    elif method == "3D":
        model_training_setup['to_device_in_model'] = True
        model_training_setup['collate_func'] = tuple_collate

    user_model_config, user_featurizer_config = process_configs(model_class, user_config)
    model_config = {**default_model_config, **user_model_config}
    featurizer_config = {**default_featurizer_config, **user_featurizer_config}

    return model_class(**model_config), featurizer_class(**featurizer_config), model_training_setup


def init_protein_seq_method(method, user_config):
    model_class, featurizer_class = {
        "CNN": (CNN, ProteinOneHotFeaturizer),
        "Transformer": (Transformer, ProteinEmbeddingFeaturizer),
        "AAD": (MLP, ProteinAADFeaturizer)
    }.get(method, (None, None))

    if model_class is None:
        raise ValueError(f"Unsupported method: {method}")

    default_model_config = {}
    default_featurizer_config = {}
    model_training_setup = {"collate_func": None, "to_device_in_model": False, "loadtime_transform": False}

    if method == "CNN":
        default_model_config = {"in_channel": 26, "input_length": ProteinOneHotFeaturizer.MAX_SEQ_PROTEIN}
        model_training_setup["loadtime_transform"] = True
    elif method == "Transformer":
        default_model_config = {
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
        model_training_setup['to_device_in_model'] = True
        model_training_setup['collate_func'] = tuple_collate
    elif method == "AAD":
        default_model_config = {"input_dim": 8420}

    user_model_config, user_featurizer_config = proess_configs(model_class, user_config)
    model_config = {**default_model_config, **user_model_config}
    featurizer_config = {**default_featurizer_config, **user_featurizer_config}

    return model_class(**model_config), featurizer_class(**featurizer_config), model_training_setup



def init_protein_pdb_method(method, user_config):
    model_class, featurizer_class = {
        "GCN": (DGL_GCN, ProteinGCNFeaturizer),
        "GCN_ESM": (DGL_GCN, ProteinGCN_ESMFeaturizer),
        "ESM": (MLP, ProteinESMFeaturizer),
        "pocket": (DTITAG, ProteinPocketFeaturizer)
    }.get(method, (None, None))

    if model_class is None:
        raise ValueError(f"Unsupported method: {method}")

    default_model_config = {}
    default_featurizer_config = {}
    model_training_setup = {"collate_func": None, "to_device_in_model": False, "loadtime_transform": False}

    if method == "GCN":
        default_model_config = {
            "in_feats" : 20,
            "output_feats" : 64
        }
        model_training_setup['collate_func'] = dgl.batch
        model_training_setup['to_device_in_model'] = True
    elif method == "GCN_ESM":
        default_model_config = {
            "in_feats": 1280,
            "output_feats": 128
        }
        model_training_setup['collate_func'] = dgl.batch
        model_training_setup['to_device_in_model'] = True
    elif method == "ESM":
        default_model_config = {
            "input_dim": 1280
        }
    elif method == "pocket":
        model_training_setup['collate_func'] = dgl.batch
        model_training_setup['to_device_in_model'] = True

    user_model_config, user_featurizer_config = process_configs(model_class, user_config)
    model_config = {**default_model_config, **user_model_config}
    featurizer_config = {**default_featurizer_config, **user_featurizer_config}

    return model_class(**model_config), featurizer_class(**featurizer_config), model_training_setup



def init_inter_layer(method, p_output_shapes, **config):
    inter_layer = None
    output_shape = None

    if method == "cat":
        default_config = {"dim": 1}
        method_config = {**default_config, **config}

        def get_cat_shape(shapes, concat_dim):
            # Handle 1D shapes (represented as integers)
            if all(isinstance(shape, int) for shape in shapes):
                return sum(shapes)

            # Handle 2D shapes (represented as tuples/lists)
            elif all(isinstance(shape, (list, tuple)) and len(shape) == 2 for shape in shapes):
                if concat_dim == 1:
                    return (sum(shape[0] for shape in shapes), shapes[0][1])
                elif concat_dim == 0:
                    return (shapes[0][0], sum(shape[1] for shape in shapes))

            else:
                raise ValueError("Inconsistent shape dimensions.")

        inter_layer = lambda *inputs: torch.cat(inputs, **method_config)
        output_shape = get_cat_shape(p_output_shapes, method_config["dim"])

    elif method == "bilinear":
        if len(p_output_shapes) != 2:
            raise ValueError("Bilinear interaction requires exactly two parent nodes.")
        default_config = {"v_dim": 64, "q_dim": 64, "h_dim": 256, "h_out": 2}
        method_config = {**default_config, **config}
        inter_layer = weight_norm(BANLayer(**method_config), name='h_mat', dim=None)
        output_shape = method_config["h_dim"]

    elif method == "fusion":
        if len(p_output_shapes) != 2:
            raise ValueError("Fusion interaction requires exactly two parent nodes.")
        dim1, dim2 = p_output_shapes
        default_config = {"mmhid2": 128}
        method_config = {**default_config, **config}
        inter_layer = BilinearFusion(dim1=dim1, dim2=dim2, mmhid1=dim1 * dim2, **method_config)
        output_shape = method_config["mmhid2"]

    elif method == "attention":
        default_config = {"num_heads": 8, "dropout": 0.1}
        method_config = {**default_config, **config}
        inter_layer = MultiHeadAttentionInteract(p_output_shapes[0], **method_config)
        output_shape = p_output_shapes[0]

    else:
        raise ValueError(f"Unknown method: {method}")

    return inter_layer, output_shape
