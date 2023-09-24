from torch.utils.data import Dataset, DataLoader
import torch

class DTIDataset(Dataset):

    def __init__(self, df, drug_transform=None, target_transform=None):
        self.drug_transform = drug_transform
        self.target_transform = target_transform
        self.labels = df["Label"].values
        self.drugs = df["Drugs"].values
        self.targets = df["Targets"].values


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        drug = self.drugs[index]
        target = self.targets[index]
        label = self.labels[index]
        if self.drug_transform is not None:
            drug = self.drug_transform(drug, 2)
        if self.target_transform is not None:
            target = self.target_transform(target, 2)
        return drug, target, label
