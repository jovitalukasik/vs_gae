import torch
import torch.nn as nn
from torch_geometric.data import Data

from models import GNNEncoder, GNNDecoder


class GNNVAE(nn.Module):
    def __init__(self, ndim, sdim, beta=.005):
        super().__init__()
        self.ndim = ndim 
        self.sdim = sdim 
        self.beta = beta
        
        self.Encoder = GNNEncoder(ndim, sdim)
        self.Decoder = GNNDecoder(ndim, sdim)
        
    def sample(self, mean, log_var, eps_scale=0.01):   #reparametrization     
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mean)
        else:
            return mean        
        
    def forward(self, batch_list):
        h_G_mean, h_G_var = self.Encoder(batch_list[0].edge_index,
                                         batch_list[0].node_atts,
                                         batch_list[0].batch)
        c = self.sample(h_G_mean, h_G_var)
        kl_loss = -0.5 * torch.sum(1 + h_G_var - h_G_mean.pow(2) - h_G_var.exp())
        recon_loss = self.Decoder(batch_list[1:], c, batch_list[0].nodes)    
        return recon_loss + self.beta*kl_loss 
    
    
    def inference(self, data, sample=False, log_var=None):
        if isinstance(data, torch.Tensor): 
            if data.size(-1) != self.sdim:
                raise Exception('Size of input is {}, must be {}'.format(data.size(0), self.ndim*2))
            if data.dim() == 1:
                mean = data.unsqueeze(0)
            else:
                mean = data
        elif isinstance(data, Data):
            if not data.__contains__('batch'):
                data.batch = torch.LongTensor([1]).to(data.edge_index.device)
            mean, log_var = self.Encoder(data.edge_index, data.node_atts, data.batch)
           
        if sample:
            c = self.sample(mean, log_var)
        else:
            c = mean
            log_var = 0
       
        edges, node_atts, edge_list = self.Decoder.inference(c)
               
        return edges, node_atts, edge_list, mean, log_var, c
    
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
