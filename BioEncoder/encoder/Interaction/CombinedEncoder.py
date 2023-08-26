import torch.nn as nn
import torch


class CombinedEncoder(nn.Module):
    def __init__(self, encoder1, encoder2, join_method="cat", head=None, mlp_hidden_layers=None):
        super(CombinedEncoder, self).__init__()
        self.encoder1 = encoder1.get_model()
        self.encoder2 = encoder2.get_model()
        self.join_method = join_method

        # Decide how to join the outputs of both encoders
        if self.join_method == "cat":
            self.join_dim = encoder1.get_output_dim() + encoder2.get_output_dim()
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

    def forward(self, x1, x2):
        out1 = self.encoder1(x1)
        out2 = self.encoder2(x2)

        # Join outputs based on method
        if self.join_method == "cat":
            out = torch.cat([out1, out2], dim=1)

        if self.mlp:
            out = self.mlp(out)
        return out
