import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from BioEncoder.task.DTI.DTILoader import *
from BioEncoder.util.data.data_processing import *


class DTITrainer:
    def __init__(self, drug_encoder, target_encoder, device='cpu', epochs=70, lr=0.001, batch_size=256, num_workers=0):
        self.device = device
        self.drug_encoder = drug_encoder
        self.protein_encoder = target_encoder
        self.model = drug_encoder.get_joined_model(target_encoder, head=1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_datasets(self, drugs, targets, y, split_frac=[1]):
        processed_drugs = self.drug_encoder.transform(drugs, 1)
        processed_targets = self.protein_encoder.transform(targets, 1)
        df = pd.DataFrame({
            'Drugs': processed_drugs,
            'Targets': processed_targets,
            'Label': y
        })
        return split(df, split_frac, shuffle=True)

    def to_device(self, v_d, v_t, y):
        if self.drug_encoder.model_training_setup["to_device_in_model"]:
            self.model.device = self.device
        else:
            v_d = v_d.float().to(self.device)

        if self.protein_encoder.model_training_setup["to_device_in_model"]:
            self.model.device = self.device
        else:
            v_t = v_t.float().to(self.device)
        y = torch.from_numpy(np.array(y)).float().to(self.device)
        return v_d, v_t, y

    def create_loader(self, dataframe, shuffle=True):
        drug_steps = self.drug_encoder.get_num_transform()
        protein_steps = self.protein_encoder.get_num_transform()
        drug_transform = self.drug_encoder.featurizer if drug_steps > 1 else None
        protein_transform = self.protein_encoder.featurizer if protein_steps > 1 else None
        data = DTIDataset(dataframe, drug_transform, protein_transform)
        params = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False
        }
        collate_func = self.drug_encoder.model_training_setup["collate_func"]
        if collate_func is not None:
            params['collate_fn'] = collate_func
        return DataLoader(data, shuffle=shuffle, **params)

    def train(self, train_df, val_df=None, test_df=None):
        print("Start training...")
        train_loader = self.create_loader(train_df)
        val_loader = self.create_loader(val_df) if val_df is not None else None
        test_loader = self.create_loader(test_df) if test_df is not None else None

        for epoch in range(self.epochs):
            self.train_one_epoch(train_loader, epoch)
            if val_loader:
                self.validate_one_epoch(val_loader, epoch)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        train_loss = 0
        for v_d, v_t, label in train_loader:
            v_d, v_t, label = self.to_device(v_d, v_t, label)
            self.optimizer.zero_grad()
            output = self.model(v_d, v_t).float().squeeze(1)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * label.size(0)

        train_loss /= len(train_loader.dataset)
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')

    def validate_one_epoch(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_d, v_t, label in val_loader:
                v_d, v_t, label = self.to_device(v_d, v_t, label)
                output = self.model(v_d, v_t).float().squeeze(1)
                loss = self.criterion(output, label)
                val_loss += loss.item() * label.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Epoch: {epoch} \tValidation Loss: {val_loss:.6f}')

