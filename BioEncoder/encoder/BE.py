from .BE_base import *
import torch.nn as nn


class DrugEncoder(Encoder):
    def __init__(self, method, **config):
        super().__init__(method, **config)
        self.input_type = BioEncoder.DRUG
        self.featurizer, self.model, self.model_training_setup = init_drug_method(self.method, self.config)


class ProteinEncoder(Encoder):
    def __init__(self, method, **config):
        super().__init__(method, **config)
        self.input_type = BioEncoder.PROT
        self.featurizer, self.model, self.model_training_setup = init_protein_method(self.method, self.config)


class ProteinPDBEncoder(Encoder):
    def __init__(self, method, **config):
        super().__init__(method, **config)
        self.input_type = BioEncoder.PROT_PDB
        self.featurizer, self.model, self.model_training_setup = init_protein_pdb_method(self.method, self.config)


class BioEncoder:
    DRUG = "drug"
    PROT = "protein"
    RNA = "rna"
    GENE = "gene"
    PROT_PDB = "protein_pdb"
    def __init__(self):
        self.nodes = {}
        self.encoders = []
        self.id_count = 0
        self.input_seq = [BioEncoder.DRUG, BioEncoder.PROT]
        self.encoder_factory = {
            BioEncoder.DRUG: DrugEncoder,
            BioEncoder.PROT: ProteinEncoder,
            BioEncoder.PROT_PDB: ProteinPDBEncoder
        }
        self.count = 0
        self.model = None

    def set_input(self, sequence):
        self.input_seq = sequence

    def create_encoder(self, encoder_type, model, name=None, **config):
        encoder = self.create_enc(encoder_type, model, config)
        encoder_node = BENode(model = encoder.get_model(), input_type=self.count, output_dim=encoder.get_output_dim())
        self.count+=1
        name = name or self._generate_name()
        self.nodes[name] = encoder_node
        self.encoders.append(encoder)
        return encoder_node

    def add(self, parent, model, output_dim=-1, name = None):
        if output_dim == -1:
            output_dim = parent.output_dim
        node = BENode(parents=[parent], model=model, output_dim = output_dim)
        name = name or self._generate_name()
        self.nodes[name] = node
        return node

    def create_enc(self, encoder_type, model, config):
        return self.encoder_factory[encoder_type](model, **config)

    def _generate_name(self):
        self.id_count += 1
        return f"node_{self.id_count}"

    def set_interaction(self, node1, node2, method, name=None, **config):
        interaction = Interaction(node1, node2, method, **config)
        interaction_node = BENode(parents = [node1, node2], model= interaction, output_dim = interaction.output_dim)
        node1.add_child(interaction_node)
        node2.add_child(interaction_node)
        name = name or self._generate_name()
        self.nodes[name] = interaction_node
        return interaction_node

    def get_node(self, name):
        return self.nodes.get(name)

    def build_model(self):
        self.model = BEModel(self.nodes, self.input_seq)
        return self.model

    def get_encoders(self):
        return self.encoders

    def get_model(self):
        return self.model



class BEModel(nn.Module):
    def __init__(self, nodes, model_input):
        super(BEModel, self).__init__()
        self.nodes = nodes
        self.input_type = model_input
        self.layers, self.input_indices_sequence = self.layered_topological_sort()

        for name, node in self.nodes.items():
            model = node.get_model()
            if model is not None and isinstance(model, nn.Module):
                self.add_module(name, model)

    def layered_topological_sort(self):
        layers = []
        processed = set()
        input_indices_sequence = []

        current_layer = [node for name, node in self.nodes.items() if node.is_root()]
        input_indices_sequence.append([[node.get_input_type()] for node in current_layer])
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

            for name, node in self.nodes.items():
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
