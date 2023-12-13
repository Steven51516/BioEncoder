import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv, GraphConv, TAGConv, GINConv, APPNPConv
from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling
from torch.autograd import Variable

class DTITAG(nn.Module):

    def __init__(self):
        super(DTITAG, self).__init__()

        self.protein_graph_conv = nn.ModuleList()
        self.protein_graph_conv.append(TAGConv(31, 64, 2))
        for i in range(3):
            self.protein_graph_conv.append(TAGConv(64, 64, 2))

        self.pooling_ligand = nn.Linear(64, 1)
        self.pooling_protein = nn.Linear(64, 1)

        self.dropout = 0.2

        self.bilstm = nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=self.dropout)

        self.fc_in = nn.Linear(8680, 4340) #1922

        self.fc_out = nn.Linear(50*31, 128)
        self.output_shape = 128
        self.device = "cpu"
    def forward(self, g):
        g = g.to(self.device)
        feature_protein = g.ndata['h']
        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g, feature_protein))
        pool_protein = GlobalAttentionPooling(self.pooling_protein)
        #31
        protein_rep = pool_protein(g, feature_protein).view(-1, 64)
        B = len(protein_rep) // 50
        protein_rep_reshaped = protein_rep.view(B, 50, -1)
        # print(protein_rep_reshaped)
        # protein_rep_reshaped = protein_rep.view(B, -1)
        # protein_rep_reshaped = self.fc_out(protein_rep_reshaped)
        return protein_rep_reshaped