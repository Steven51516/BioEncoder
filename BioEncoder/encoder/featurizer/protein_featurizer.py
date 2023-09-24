from sklearn.preprocessing import OneHotEncoder
from BioEncoder.util.biochem.shared import BPEEncoder
from .featurizer import Featurizer
from BioEncoder.util.biochem.protein import *
import numpy as np


class ProteinOneHotFeaturizer(Featurizer):
    MAX_SEQ_PROTEIN = 1000

    def __init__(self):
        super(ProteinOneHotFeaturizer, self).__init__()
        amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
                      'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']
        self.onehot_enc = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
        self.amino_char = amino_char

    def step1(self, x):
        temp = list(x.upper())
        temp = [i if i in self.amino_char else '?' for i in temp]
        if len(temp) < ProteinOneHotFeaturizer.MAX_SEQ_PROTEIN:
            temp = temp + ['?'] * (ProteinOneHotFeaturizer.MAX_SEQ_PROTEIN - len(temp))
        else:
            temp = temp[:ProteinOneHotFeaturizer.MAX_SEQ_PROTEIN]
        return temp

    def step2(self, x):
        return self.onehot_enc.transform(np.array(x).reshape(-1, 1)).toarray().T

    def transform(self, x):
        x_step1 = self.step1(x)
        return self.step2(x_step1)


class TargetAACFeaturizer(Featurizer):

    def __init__(self):
        super(TargetAACFeaturizer, self).__init__()

    def transform(self, s):
        try:
            features = CalculateAADipeptideComposition(s)
        except:
            print('AAC fingerprint not working for smiles: ' + s + ' convert to 0 vectors')
            features = np.zeros((8420,))
        return np.array(features)


class ProteinEmbeddingFeaturizer(Featurizer):
    def __init__(self, vocab_path="BioEncoder/util/biochem/shared/ESPF/protein_codes_uniprot_2000.txt",
                 subword_map_path="BioEncoder/util/biochem/shared/ESPF/subword_units_map_uniprot_2000.csv", max_d=545):
        super(ProteinEmbeddingFeaturizer, self).__init__()
        self.drug_encoder = BPEEncoder(vocab_path, subword_map_path)
        self.max_d = max_d

    def transform(self, x):
        return self.drug_encoder.encode(x, self.max_d)


class ProteinAADFeaturizer(Featurizer):
    def transform(self, x):
        return CalculateAADipeptideComposition(x)


from BioEncoder.util.biochem.protein.prot_graph import *
class ProteinGCNFeaturizer(Featurizer):
    def __init__(self, virtual_nodes=False):
        super(ProteinGCNFeaturizer, self).__init__()
        self.virtual_nodes = virtual_nodes
    def transform(self, x):
        x = create_prot_dgl_graph(x)
        if self.virtual_nodes:
            actual_node_feats = x.ndata.pop('h')
            num_actual_nodes = actual_node_feats.shape[0]
            num_virtual_nodes = 604 - num_actual_nodes
            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
            x.ndata['h'] = actual_node_feats
            virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 20), torch.ones(num_virtual_nodes, 1)), 1)
            x.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
            x = x.add_self_loop()
        return x


class ProteinGCN_ESMFeaturizer(Featurizer):
    def transform(self, x):
        x = create_prot_esm_dgl_graph(x)
        return x


class ProteinESMFeaturizer(Featurizer):
    def transform(self, x):
        x = create_prot_esm_embedding(x)
        return x




