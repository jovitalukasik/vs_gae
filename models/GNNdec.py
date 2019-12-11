import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli, Categorical

from models.GNNenc import GNNLayer
from utils import edges2index



class NodeEmbUpd(nn.Module):
    def __init__(self,
                 ndim,
                 num_layers=2,
                 node_dropout=.0,
                 dropout=.0):
        super().__init__()
        self.ndim = ndim
        self.num_layers = num_layers
        self.dropout = dropout
        self.Dropout = nn.Dropout(dropout)
        self.GNNLayers = nn.ModuleList([GNNLayer(ndim) for _ in range(num_layers)])
        
    def forward(self, h, edge_index):
        edge_index = torch.cat([edge_index,
                                torch.index_select(edge_index, 0, torch.tensor([1, 0]).to(h.device))
                               ], dim=1
                              )
        
        for layer in self.GNNLayers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(edge_index, h)
        return h
    
    
    
class GraphAggr(nn.Module):
    def __init__(self, ndim, sdim, aggr='gsum'):
        super().__init__()
        self.ndim = ndim
        self.sdim = sdim
        self.aggr = aggr
        self.f_m = nn.Linear(ndim, sdim)
        if aggr == 'gsum':
            self.g_m = nn.Linear(ndim, 1)
            self.sigm = nn.Sigmoid()

    def forward(self, h, idx):
        if self.aggr == 'mean':
            h = self.f_m(h).view(-1, idx, self.sdim)
            return torch.mean(h, 1)
        elif self.aggr == 'gsum':
            h_vG = self.f_m(h)
            g_vG = self.sigm(self.g_m(h))
            h = torch.mul(h_vG, g_vG).view(-1, idx, self.sdim)        
            return torch.sum(h, 1)
        
        
        
class GraphEmbed(nn.Module):
    def __init__(self, ndim, sdim):
        super().__init__()
        self.ndim = ndim
        self.sdim = sdim
        self.NodeEmb = NodeEmbUpd(ndim)
        self.GraphEmb = GraphAggr(ndim, sdim)
        self.GraphEmb_init = GraphAggr(ndim, sdim)
        
    def forward(self, h, edge_index):
        idx = h.size(1)
        h = h.view(-1, self.ndim)
        if idx == 1:
            return h.unsqueeze(1), self.GraphEmb.f_m(h), self.GraphEmb_init.f_m(h)
        else:
            h = self.NodeEmb(h, edge_index)
            h_G = self.GraphEmb(h, idx)
            h_G_init = self.GraphEmb_init(h, idx)
            return h.view(-1, idx, self.ndim), h_G, h_G_init
        

class NodeAdd(nn.Module):
    def __init__(self, sdim, num_node_atts=5):
        super().__init__()
        self.sdim = sdim
        self.f_an = nn.Linear(sdim*2, sdim)
        self.f_an_2 = nn.Linear(sdim, num_node_atts)
        
    def forward(self, h_G, c):
        s = self.f_an(torch.cat([h_G, c], 1))        
        return self.f_an_2(F.relu(s))  
        
        
        
class NodeInit(nn.Module):
    def __init__(self, ndim, sdim, num_node_atts=5):
        super().__init__()
        self.ndim = ndim
        self.sdim = sdim
        self.NodeInits = nn.Embedding(num_node_atts, ndim)
        self.f_init = nn.Linear(ndim+sdim*2, ndim+sdim)
        self.f_init_2 = nn.Linear(ndim+sdim, ndim)
        self.f_start = nn.Linear(ndim+sdim, ndim+sdim) 
        self.f_start_2 = nn.Linear(ndim+sdim, ndim)
        
    def forward(self, h_G_init, node_atts, c):
        e = self.NodeInits(node_atts)
        if isinstance(h_G_init, str):
            h_inp = self.f_start(torch.cat([e, c], 1))
            return self.f_start_2(F.relu(h_inp))
        h_v = self.f_init(torch.cat([e, h_G_init, c], 1))
        return self.f_init_2(F.relu(h_v))
    
    
    
class Nodes(nn.Module): 
    def __init__(self, ndim, sdim):
        super().__init__()
        self.ndim = ndim
        self.sdim = sdim
        self.f_s_1 = nn.Linear(ndim*2+sdim*2, ndim+sdim)
        self.f_s_2 = nn.Linear(ndim+sdim, 1)
    
    def forward(self, h, h_v, h_G, c):
        idx = h.size(1)
        s = self.f_s_1(torch.cat([h.view(-1, self.ndim),
                                  h_v.unsqueeze(1).repeat(1, idx, 1).view(-1, self.ndim),
                                  h_G.repeat(idx, 1),
                                  c.repeat(idx, 1)], dim=1)) 
        return self.f_s_2(F.relu(s)).view(-1, idx)
    
    
    
    
class Generator(nn.Module):
    def __init__(self, ndim, sdim, alpha=.5, stop=20):
        super().__init__()
        self.ndim = ndim
        self.sdim = sdim
        self.alpha = alpha
        self.prop = GraphEmbed(ndim, sdim) 
        self.nodeAdd = NodeAdd(sdim) 
        self.nodeInit = NodeInit(ndim, sdim) 
        self.nodes = Nodes(ndim, sdim)
        self.node_criterion = torch.nn.CrossEntropyLoss(reduction='none') 
        self.edge_criterion = torch.nn.BCEWithLogitsLoss(reduction='none') 
        self.stop = stop

        
    def forward(self, h, c, edge_index, node_atts, edges):
        h, h_G, h_G_init = self.prop(h, edge_index) 
        node_score = self.nodeAdd(h_G, c)
        node_loss = self.node_criterion(node_score, node_atts)    
        h_v = self.nodeInit(h_G_init, node_atts, c) 
    
        if h.size(1) == 1: 
            h = torch.cat([h, h_v.unsqueeze(1)], 1)
            return h, 2*(1-self.alpha)*node_loss
        
        edge_score = self.nodes(h, h_v, h_G, c) 
        edge_loss = torch.mean(self.edge_criterion(edge_score, edges), 1)
        h = torch.cat([h, h_v.unsqueeze(1)], 1)
        return h, 2*((1-self.alpha)*node_loss + self.alpha*edge_loss)
    
    
    def inference(self, h, c, edge_index):
        h, h_G, h_G_init = self.prop(h, edge_index)
        node_logit = self.nodeAdd(h_G, c)
        node_atts = Categorical(logits=node_logit).sample().long()
        non_zero = (node_atts != 0)
        h_v = self.nodeInit(h_G_init, node_atts, c)
        
        if h.size(1) == 1:
            edges = torch.ones_like(node_atts).unsqueeze(1)
            h = torch.cat([h, h_v.unsqueeze(1)], 1)
            return h, node_atts.unsqueeze(1), edges, non_zero
        
        edge_logit = self.nodes(h, h_v, h_G, c)
        edges = Bernoulli(logits=edge_logit).sample().long()
        h = torch.cat([h, h_v.unsqueeze(1)], 1)
        
        if h.size(1) >= self.stop:
            print('stop generating, since max size ({}) was reached'.format(self.stop))
            non_zero = torch.zeros_like(non_zero)
             
        
        return h, node_atts.unsqueeze(1), edges, non_zero

    
    
    
    
class GNNDecoder(nn.Module):  
    def __init__(self, ndim, sdim):
        super().__init__()
        self.ndim = ndim
        self.sdim = sdim
        self.generator = Generator(ndim, sdim)

    def forward(self, batch_list, c, nodes):
        h = self.generator.nodeInit('start', torch.ones_like(batch_list[1].node_atts), c).unsqueeze(1) 
        loss_list = torch.Tensor().to(c.device)
        edge_index = 0
        for batch in batch_list:
            h, loss = self.generator(h,
                                     c,
                                     edge_index,
                                     batch.node_atts,
                                     batch.edges
                                    )
            loss_list = torch.cat([loss_list, loss.unsqueeze(1)], 1)
            edge_index = batch.edge_index
        return torch.mean(torch.sum(torch.mul(loss_list, nodes), 1), 0)

    
    def inference(self, c):
        batch_size = c.size(0)
        h = self.generator.nodeInit('start', torch.ones(batch_size, dtype=torch.long).to(c.device), c).unsqueeze(1)        
        h, node_atts, edges, non_zeros = self.generator.inference(h, c, None)
        graph = node_atts.clone()
        node_atts = torch.cat([edges, node_atts], 1)
        num_zeros = (non_zeros == 0).sum().item()
        while num_zeros < batch_size:
            edge_index = edges2index(edges)
            h, node_atts_new, edges_new, non_zero = self.generator.inference(h, c, edge_index)
            graph = torch.cat([graph, node_atts_new, edges_new], 1)
            node_atts = torch.cat([node_atts, node_atts_new], 1)
            edges = torch.cat([edges, edges_new], 1)
            non_zeros = torch.mul(non_zeros, non_zero)
            num_zeros = (non_zeros == 0).sum().item()
        return graph, node_atts.view(batch_size, -1), edges2index(edges, finish=True)
