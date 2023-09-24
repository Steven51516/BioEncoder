from torch.utils.data import Dataset, DataLoader
import torch
import copy

class BEDataset(Dataset):
    def __init__(self, df, encoders):
        # Get all columns except the last one as features
        self.features = df.iloc[:, :-1].values
        # Get the last column as labels
        self.labels = df.iloc[:, -1].values
        # Store the transformations
        self.encoders = encoders

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row_features = copy.deepcopy(self.features[index])
        label = self.labels[index]
        for i, encoder in enumerate(self.encoders):
            if encoder.get_num_transform() > 1:
                row_features[i] = encoder.featurizer(row_features[i], 2)
        return *row_features, label

