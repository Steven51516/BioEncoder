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
    def __init__(self, vocab_path="BioEncoder/util/data/shared/ESPF/protein_codes_uniprot_2000.txt",
                 subword_map_path="BioEncoder/util/data/shared/ESPF/subword_units_map_uniprot_2000.csv", max_d=545):
        super(ProteinEmbeddingFeaturizer, self).__init__()
        self.drug_encoder = BPEEncoder(vocab_path, subword_map_path)
        self.max_d = max_d

    def transform(self, x):
        return self.drug_encoder.encode(x, self.max_d)


