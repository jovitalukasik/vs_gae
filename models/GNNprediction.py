import torch
import torch.nn as nn
from models.GNNenc import NodeEmb, GraphEmb, GetAcc



class GNNpred(nn.Module):
    def __init__(self, 
                 ndim,
                 sdim,
                 num_gnn_layers=2,
                 node_dropout=.0,
                 g_aggr='gsum',
                 dropout=.0):
        super().__init__()
        self.NodeEmb = NodeEmb(ndim,
                               num_gnn_layers,
                               node_dropout,
                               dropout)
        self.GraphEmb_mean = GraphEmb(ndim, sdim, g_aggr)
        self.Accuracy = GetAcc(sdim, num_layers=4, dropout=.0)
        
    def forward(self, edge_index, node_atts, batch):
        h = self.NodeEmb(edge_index, node_atts)
        h_G_mean = self.GraphEmb_mean(h, batch)
        acc = self.Accuracy(h_G_mean)
        return acc

    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
    

    