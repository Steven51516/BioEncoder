import torch.nn as nn
import torch.nn.functional as F
import torch


class BilinearFusion(nn.Module):
    def __init__(self, skip=False,
                 gate1=True, gate2=True, gated_fusion=True,
                 dim1=32, dim2=32,
                 scale_dim1=1, scale_dim2=1,
                 mmhid1=4096, mmhid2=16, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.gate1 = gate1
        self.gate2 = gate2
        self.gated_fusion = gated_fusion

        dim1, dim2 = int(dim1 / scale_dim1), int(dim2 / scale_dim2)

        self.linear_h1 = nn.Linear(dim1, dim1)
        self.linear_z1 = nn.Linear(dim1 + dim2, dim1)  # Assuming input dim is sum of vec1 and vec2
        if self.gate1:
            self.linear_o1 = nn.Sequential(
                nn.Linear(dim1, dim1),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

        self.linear_h2 = nn.Linear(dim2, dim2)
        self.linear_z2 = nn.Linear(dim1 + dim2, dim2)
        if self.gate2:
            self.linear_o2 = nn.Sequential(
                nn.Linear(dim2, dim2),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

        self.post_fusion_dropout = nn.Dropout(dropout_rate)
        self.encoder1 = nn.Sequential(
            nn.Linear(mmhid1, mmhid1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid1, mmhid1),
            nn.ReLU(),
            nn.Linear(mmhid1, mmhid2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vec1, vec2):
        if self.gate1:
            h1 = F.relu(self.linear_h1(vec1))
            z1 = self.linear_z1(torch.cat([vec1, vec2], dim=1))
            o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        else:
            o1 = F.relu(self.linear_h1(vec1))

        if self.gate2:
            h2 = F.relu(self.linear_h2(vec2))
            z2 = self.linear_z2(torch.cat([vec1, vec2], dim=1))
            o2 = self.linear_o2(torch.sigmoid(z2) * h2)
        else:
            o2 = F.relu(self.linear_h2(vec2))

        if self.gated_fusion:
            o1_m = o1.unsqueeze(2)
            o2_m = o2.unsqueeze(1)

            o12 = (o1_m @ o2_m).view(o1_m.shape[0], -1)  # Batch matrix multiply and flatten
            out = self.post_fusion_dropout(o12)
            out = self.encoder1(out)
            if self.skip:
                out = torch.cat([out, o1, o2], dim=1)
        else:
            out = torch.cat([o1, o2], dim=1)
        out = self.encoder2(out)
        return out

