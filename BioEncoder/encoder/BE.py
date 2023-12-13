from .BE_base import *
from collections import OrderedDict
import torch.nn as nn


class DrugEncoder(Encoder):
    def __init__(self, method, **config):
        super().__init__(method, **config)
        self.input_type = BioEncoder.DRUG
        self.model, self.featurizer, self.model_training_setup = init_drug_method(self.method, config)


class ProteinSEQEncoder(Encoder):
    def __init__(self, method, **config):
        super().__init__(method, **config)
        self.input_type = BioEncoder.PROT
        self.model, self.featurizer, self.model_training_setup = init_protein_seq_method(self.method, config)


class ProteinPDBEncoder(Encoder):
    def __init__(self, method, **config):
        super().__init__(method, **config)
        self.input_type = BioEncoder.PROT_PDB
        self.model, self.featurizer, self.model_training_setup = init_protein_pdb_method(self.method, config)


class BioEncoder:
    """
    BioEncoder class for initializing and managing different types of encoders
    (Drug, Protein Sequence, Protein PDB) and setting up interactions between them.

    Attributes:
        nodes (list): List of nodes representing individual models or interactions.
        encoders_dic (OrderedDict): Dictionary mapping encoders to their corresponding nodes.
        encoder_factory (dict): Factory for creating encoder instances.
        model (nn.Module): Compiled model, created after building the encoders and interactions.

    Methods:
        init_drug_encoder(model, **config): Initializes a drug encoder.
        init_prot_encoder(model, pdb=False, **config): Initializes a protein encoder, with optional PDB.
        init_encoder(encoder_type, model, **config): Generic initializer for any encoder.
        set_interaction(nodes, method, **config): Sets up an interaction between given nodes.
        register_interaction(method, interaction_layer, output_shape): Register a custom interaction method.
        cat(nodes, **config): Convenience method for concatenation interaction.
        build_model(): Builds the final model from the initialized nodes.
        get_encoders(): Retrieves the list of initialized encoders.
        get_model(): Retrieves the compiled model.
    """

    DRUG = "DRUG"
    PROT_SEQ = "PROT_SEQ"
    PROT_PDB = "PROT_PDB"

    def __init__(self):
        self.nodes = []
        self.encoders_dic = OrderedDict()
        self.encoder_factory = {
            BioEncoder.DRUG: DrugEncoder,
            BioEncoder.PROT_SEQ: ProteinSEQEncoder,
            BioEncoder.PROT_PDB: ProteinPDBEncoder
        }
        self.model = None

        self.custom_interactions = {}

    def init_drug_encoder(self, model, **config):
        return self.init_encoder(BioEncoder.DRUG, model, **config)

    def init_prot_encoder(self, model, pdb=False, **config):
        if pdb:
            return self.init_encoder(BioEncoder.PROT_PDB, model, **config)
        return self.init_encoder(BioEncoder.PROT_SEQ, model, **config)

    def init_encoder(self, encoder_type, model, **config):
        encoder = self.encoder_factory[encoder_type](model, **config)
        encoder_node = BENode(model=encoder.get_model(), root_idx=len(self.encoders_dic),
                              output_shape=encoder.get_output_shape())
        self.nodes.append(encoder_node)
        self.encoders_dic[encoder] = encoder_node
        return encoder

    def set_interaction(self, nodes, method, **config):
        nodes = [self.encoders_dic[node] if isinstance(node, Encoder) else node for node in nodes]
        inter_model = Interaction(nodes, method, **config)
        inter_node = BENode(parents=nodes, model=inter_model, output_shape=inter_model.get_output_shape())
        self.nodes.append(inter_node)
        return inter_node

    def cat(self, nodes, **config):
        return self.set_interaction(nodes, "cat", **config)

    def build_model(self):
        self.model = BEModel(self.nodes)
        return self.model

    def get_encoders(self):
        return list(self.encoders_dic.keys())

    def get_model(self):
        return self.model


class BEModel(nn.Module):
    def __init__(self, nodes):
        super(BEModel, self).__init__()
        self.nodes = nodes
        self.layers, self.input_indices_sequence = self.layered_topological_sort()

        for i, node in enumerate(self.nodes):
            model = node.get_model()
            if model is not None and isinstance(model, nn.Module):
                setattr(self, f"module_{i}", model)

    def layered_topological_sort(self):
        layers = []
        processed = set()
        input_indices_sequence = []

        current_layer = [node for node in self.nodes if node.is_root()]
        input_indices_sequence.append([[node.get_root_idx()] for node in current_layer])
        indices = {}
        processed.update(current_layer)
        current_index = len(input_indices_sequence[0])
        while current_layer:
            layers.append(current_layer)
            layer_input_indices = []
            next_layer = []

            for node in current_layer:
                indices[node] = current_index
                current_index += 1

            for node in self.nodes:
                if node not in processed and all(parent in processed for parent in node.get_parents()):
                    layer_input_indices.append([indices[parent] for parent in node.get_parents()])
                    next_layer.append(node)
            input_indices_sequence.append(layer_input_indices)
            processed.update(next_layer)
            current_layer = next_layer

        return layers, input_indices_sequence

    def forward(self, *x):
        current_outputs = list(x)
        for layer, layer_input_indices in zip(self.layers, self.input_indices_sequence):
            next_outputs = []
            for node, input_indices in zip(layer, layer_input_indices):
                inputs = [current_outputs[index] for index in input_indices]
                if len(inputs) == 1:
                    if isinstance(inputs[0], tuple):
                        next_outputs.append(node.get_model()(*inputs[0]))
                    else:
                        next_outputs.append(node.get_model()(inputs[0]))
                else:
                    next_outputs.append(node.get_model()(*inputs))
            current_outputs.extend(next_outputs)
        return current_outputs[-1]
