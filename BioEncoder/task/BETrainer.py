import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch
from .BELoader import BEDataset
from BioEncoder.util.data.data_processing import *
from BioEncoder.util.data.collate_func import *


class BETrainer:
    def __init__(self, BE, device='cpu', epochs=100, lr=0.001, batch_size=256,
                 num_workers=0, scheduler=None):
        self.device = device
        self.model = BE.get_model()
        self.encoders = BE.get_encoders()
        for encoder in self.encoders:
            if encoder.model_training_setup["to_device_in_model"]:
                encoder.model.device = self.device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        if scheduler == "exp":
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        else:
            self.scheduler = None
        self.model.to(self.device)

    def prepare_datasets(self, inputs, input_types, y, split_frac=[1]):
        # Check if the number of inputs and input_types are consistent
        assert len(inputs) == len(input_types), "Inconsistent number of inputs and input_types"

        data_dict = {}  # Using a dictionary to store processed data with keys as column names

        # Create a map of input type to its corresponding data for efficient lookup
        input_map = dict(zip(input_types, inputs))

        # Process data with respective encoders based on the encoder's input type
        for idx, encoder in enumerate(self.encoders):
            input_type = encoder.input_type
            assert input_type in input_map, f"Data for encoder's input type {input_type} not found in inputs"
            input_data = input_map[input_type]
            if encoder.model_training_setup["loadtime_transform"]:
                processed_data = encoder.transform(input_data, mode="initial")
            else:
                processed_data = encoder.transform(input_data)
            data_dict[f'Input_{idx}'] = processed_data

        data_dict['Label'] = y

        df = pd.DataFrame(data_dict)

        return split(df, split_frac, shuffle=True)

    def to_device(self, *inputs_and_label):
        inputs_and_label = list(inputs_and_label)
        for idx, encoder in enumerate(self.encoders):
            if not encoder.model_training_setup["to_device_in_model"]:
                inputs_and_label[idx] = inputs_and_label[idx].float().to(self.device)
        inputs_and_label[-1] = torch.from_numpy(np.array(inputs_and_label[-1])).float().to(self.device)
        return tuple(inputs_and_label)

    def create_loader(self, dataframe, shuffle=True):
        data = BEDataset(dataframe, self.encoders)
        params = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False
        }
        collate_funcs = [encoder.model_training_setup["collate_func"] for encoder in self.encoders]
        from functools import partial
        if collate_funcs:
            params['collate_fn'] = partial(get_collate, collate_func=collate_funcs)
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
            if self.scheduler is not None:
                self.scheduler.step()

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        train_loss = 0
        for *inputs, label in train_loader:
            inputs_and_label = self.to_device(*inputs, label)
            self.optimizer.zero_grad()
            output = self.model(*inputs_and_label[:-1]).float().squeeze(1)
            loss = self.criterion(output, inputs_and_label[-1])
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * inputs_and_label[-1].size(0)
        train_loss /= len(train_loader.dataset)
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')

    def validate_one_epoch(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for *inputs, label in val_loader:
                inputs_and_label = self.to_device(*inputs, label)
                output = self.model(*inputs_and_label[:-1]).float().squeeze(1)
                loss = self.criterion(output, inputs_and_label[-1])
                val_loss += loss.item() * inputs_and_label[-1].size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Epoch: {epoch} \tValidation Loss: {val_loss:.6f}')
