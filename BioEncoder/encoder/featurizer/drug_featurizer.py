from BioEncoder.util.biochem.shared import BPEEncoder
from BioEncoder.util.biochem.drug import *
from .featurizer import Featurizer
from sklearn.preprocessing import OneHotEncoder
from functools import partial
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np


class DrugOneHotFeaturizer(Featurizer):
    MAX_SEQ_DRUG = 100

    def __init__(self):
        super(DrugOneHotFeaturizer, self).__init__()
        smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
                       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
                       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
                       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
                       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']
        self.onehot_enc = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
        self.smiles_char = smiles_char

    def step1(self, x):
        temp = list(x)
        temp = [i if i in self.smiles_char else '?' for i in temp]
        if len(temp) < DrugOneHotFeaturizer.MAX_SEQ_DRUG:
            temp = temp + ['?'] * (DrugOneHotFeaturizer.MAX_SEQ_DRUG - len(temp))
        else:
            temp = temp[:DrugOneHotFeaturizer.MAX_SEQ_DRUG]
        return temp

    def step2(self, x):
        return self.onehot_enc.transform(np.array(x).reshape(-1, 1)).toarray().T

    def transform(self, x):
        x_step1 = self.step1(x)
        return self.step2(x_step1)


class GCNGraphFeaturizer(Featurizer):
    def __init__(self):
        super().__init__()
        self.node_featurizer = CanonicalAtomFeaturizer()
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.transform_func = partial(smile_to_bigraph,
                                      node_featurizer=self.node_featurizer,
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=True)

    def transform(self, x):
        return self.transform_func(x)


class MorganFeaturizer(Featurizer):
    def __init__(self, radius=2, nbits=1024):
        super().__init__()
        self.radius = radius
        self.nBits = nbits

    def transform(self, s):
        try:
            mol = Chem.MolFromSmiles(s)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except:
            print('rdkit not found this smiles for morgan: ' + s + ' convert to all 0 features')
            features = np.zeros((self.nBits,))
        return features


class DrugEmbeddingFeaturizer(Featurizer):
    def __init__(self, vocab_path="BioEncoder/util/biochem/shared/ESPF/drug_codes_chembl_freq_1500.txt",
                 subword_map_path="BioEncoder/util/biochem/shared/ESPF/subword_units_map_chembl_freq_1500.csv",
                 max_d=50):
        super(DrugEmbeddingFeaturizer, self).__init__()
        self.drug_encoder = BPEEncoder(vocab_path, subword_map_path)
        self.max_d = max_d

    def transform(self, x):
        return self.drug_encoder.encode(x, self.max_d)
