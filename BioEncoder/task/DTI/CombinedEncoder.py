import torch.nn as nn
import torch
import torch.nn.functional as F


class CombinedEncoder(nn.Module):
    def __init__(self, encoder1, encoder2, join_method="cat", head=None, mlp_hidden_layers=None):
        super(CombinedEncoder, self).__init__()
        self.encoder1 = encoder1.get_model()
        self.encoder2 = encoder2.get_model()
        self.join_method = join_method

        # Decide how to join the outputs of both encoders
        if self.join_method == "cat":
            self.join_dim = encoder1.get_output_dim() + encoder2.get_output_dim()
        elif self.join_method == "dot":
            self.join_dim = self.get_join_dim(encoder1.get_output_dim(), encoder2.get_output_dim())
        else:
            raise ValueError(f"Join method not recognized")

        if head:
            if mlp_hidden_layers is None:
                mlp_hidden_layers = [1024, 718, 512]
            mlp_layers = []
            input_dim = self.join_dim
            for hidden_dim in mlp_hidden_layers:
                mlp_layers.append(nn.Linear(input_dim, hidden_dim))
                mlp_layers.append(nn.ReLU())  # activation function
                input_dim = hidden_dim
            if head:
                mlp_layers.append(nn.Linear(input_dim, head))
            self.mlp = nn.Sequential(*mlp_layers)
        else:
            self.mlp = None

    def get_join_dim(self, dim1, dim2):
        # Create a dummy tensor
        dummy_out1 = torch.zeros(1, dim1)
        dummy_out2 = torch.zeros(1, dim2)

        # Simulate the forward method's operations on the dummy tensor
        dummy_tensor = torch.bmm(dummy_out1.unsqueeze(2), dummy_out2.unsqueeze(1))
        # Return the flattened size
        return dummy_tensor.view(dummy_tensor.size(0), -1).size(1)

    def forward(self, x1, x2):
        out1 = self.encoder1(x1)
        out2 = self.encoder2(x2)

        if self.join_method == "cat":
            out = torch.cat([out1, out2], dim=1)
        elif self.join_method == "dot":
            out = torch.bmm(out1.unsqueeze(2), out2.unsqueeze(1))  # This results in shape [batch_size, dim1, dim2]
            out = out.view(out.size(0), -1)
        # elif self.join_method == "bilinear":
        #     from torch.nn.utils.weight_norm import weight_norm
        #     from .bilinear import *
        #     self.bcn = weight_norm(
        #         BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
        #         name='h_mat', dim=None)

        if self.mlp:
            out = self.mlp(out)
        return out
