import torch
import h5py
import numpy as np
from numpy.random import shuffle as shffle
from torch_geometric.data import Data, DataLoader
import pdb
from torch_geometric.utils import to_dense_adj

def edges2index(edges, finish=False):
    device = edges.device
    batch_size, size = edges.size()
    edge_index = torch.LongTensor(2, 0).to(device)
    num_nodes = int(np.sqrt(2*size+1/4)+.5)
    for idx, batch_edge in enumerate(edges):
        trans = idx*num_nodes
        if finish:
            trans = 0
        i = np.inf
        j = 0
        for k, edge in enumerate(batch_edge):
            if j >= i:
                j -= i
                
            if edge.item() == 0:
                j += 1
                i = int(np.ceil((np.sqrt(2*k+9/4)-.5)))
            else:
                i = int(np.ceil((np.sqrt(2*k+9/4)-.5)))
                edge_index = torch.cat([edge_index, torch.LongTensor([[j+trans], [i+trans]]).to(device)], 1)
                j += 1
    return edge_index


def batch2graph(graphs, stop=20):
    graphs = graphs.cpu().numpy()
    output = list()
    for graph in graphs:
        graph = list(graph)
        node_att = graph[0]
        node_atts = [1, node_att]
        edges = list([1])
        idx = 1
        run = 3
        while node_att != 0:
            if run >= stop:
                break
            node_att = graph[idx]
            node_atts += [node_att]
            edges += graph[idx+1:idx+run]
            idx += run
            run += 1
        edge_index = edges2index(torch.tensor(edges).unsqueeze(0))
        output.append((torch.tensor(node_atts), edge_index))
    return output
    
    
    
def gauss(n):
    n = int(n)
    return n*(n+1)//2

def get_edges(edge_list, max_num_nodes):
    edge_list_sorted = np.array(sorted(edge_list, key=lambda edge: tuple(edge[::-1])))
    edges = np.zeros(gauss(max_num_nodes-1), dtype=np.float32)
    for edge in edge_list_sorted:
        idx = edge[0] + gauss(edge[1]-1)
        edges[idx] = 1.
    return edges, edge_list_sorted

def get_edge_list_list(edge_list_sorted, max_num_nodes):
    edge_list_list = [np.array([[0, 1]]) for _ in range(max_num_nodes-1)]
    dst = 1
    for idx, edge in enumerate(edge_list_sorted):
        if edge[1] != dst:
            dst = edge[1]
            edge_list_list[dst-2] = edge_list_sorted[:idx]
    edge_list_list[dst-1] = edge_list_sorted
    return edge_list_list


def prep_data(data_path, max_num_nodes=None, training=False, aggr='sum', device='cpu'):
    if data_path[-4:] == '.pth':
        if not training:
            return torch.load(data_path) 
        else:
            device = torch.device(device)
            data_list = list()
            data=torch.load(data_path)
            for graph in data:
                edges, edge_list =get_edges(graph.edge_index.numpy().transpose(),max_num_nodes)
                edge_list_list = get_edge_list_list(edge_list, max_num_nodes)
                node_atts=graph.node_atts.numpy()
                num_nodes = node_atts.size
                node_atts_padded = np.ones(max_num_nodes, dtype=np.int32)
                node_atts_padded[:num_nodes-1] = node_atts[1:]
                nodes = np.zeros(max_num_nodes-1, dtype=np.float32)
                nodes[:num_nodes-1] = 1
                if aggr == 'mean':
                    nodes /= num_nodes
                acc = graph.acc.numpy().item()
#                 test_acc=graph.test_acc.numpy().item()
#                 training_time=graph.training_time.numpy().item()
                data = Data(edge_index=torch.LongTensor(np.transpose(edge_list)).to(device),
                            num_nodes=num_nodes,
                            node_atts=torch.LongTensor(node_atts).to(device),
                            acc = torch.tensor([acc]).to(device),
#                             test_acc=torch.tensor([test_acc]).to(device),
#                             training_time=torch.tensor([training_time]).to(device),
                            nodes=torch.tensor(nodes).unsqueeze(0).to(device),
                           )

                data_full = [data]
                for idx in range(max_num_nodes-1):
                    num_nodes = idx + 2
                    data = Data(edge_index=torch.LongTensor(np.transpose(edge_list_list[idx])).to(device),
                            num_nodes=num_nodes,
                            node_atts=torch.LongTensor([node_atts_padded[idx]]).to(device),
                            edges=torch.tensor(edges[gauss(num_nodes-2):gauss(num_nodes-1)]).unsqueeze(0).to(device)
                           )
                    data_full.append(data)
                data_list.append(tuple(data_full))   
    else:            
        f = h5py.File(data_path, 'r')    
        device = torch.device(device)
        data_list = list()
        for graph in f.values():
            if not training:
                edge_list = graph['edgelist'][()]
                node_atts = graph['node_attr'][()] + 2
                acc = graph['val_acc'][()]
                test_acc=graph['test_acc'][()]
                training_time=graph['training_time'][()]
                num_nodes = node_atts.size
                try:
                    edit=graph['edit_dist'][()]
                    data = Data(edge_index=torch.LongTensor(np.transpose(edge_list)).to(device),
                        num_nodes=num_nodes,
                        node_atts=torch.LongTensor(node_atts).to(device),
                        acc = torch.tensor([acc]).to(device),
                        test_acc=torch.tensor([test_acc]).to(device),
                        training_time=torch.tensor([training_time]).to(device),
                        edit=torch.tensor([edit]).to(device),
                       )
                except:
                    data = Data(edge_index=torch.LongTensor(np.transpose(edge_list)).to(device),
                        num_nodes=num_nodes,
                        node_atts=torch.LongTensor(node_atts).to(device),
                        acc = torch.tensor([acc]).to(device),
                        test_acc=torch.tensor([test_acc]).to(device),
                        training_time=torch.tensor([training_time]).to(device),
                       )
                data_list.append(data)
                continue
            else: 
                edges, edge_list = get_edges(graph['edgelist'][()], max_num_nodes)
                edge_list_list = get_edge_list_list(edge_list, max_num_nodes)
                node_atts = graph['node_attr'][()] + 2
                num_nodes = node_atts.size
                node_atts_padded = np.ones(max_num_nodes, dtype=np.int32)
                node_atts_padded[:num_nodes-1] = node_atts[1:]
                nodes = np.zeros(max_num_nodes-1, dtype=np.float32)
                nodes[:num_nodes-1] = 1
                if aggr == 'mean':
                    nodes /= num_nodes
                acc = graph['val_acc'][()]
                test_acc=graph['test_acc'][()]
                training_time=graph['training_time'][()]
                data = Data(edge_index=torch.LongTensor(np.transpose(edge_list)).to(device),
                            num_nodes=num_nodes,
                            node_atts=torch.LongTensor(node_atts).to(device),
                            acc = torch.tensor([acc]).to(device),
                            test_acc=torch.tensor([test_acc]).to(device),
                            training_time=torch.tensor([training_time]).to(device),
                            nodes=torch.tensor(nodes).unsqueeze(0).to(device),
                           )

                data_full = [data]
                for idx in range(max_num_nodes-1):
                    num_nodes = idx + 2
                    data = Data(edge_index=torch.LongTensor(np.transpose(edge_list_list[idx])).to(device),
                            num_nodes=num_nodes,
                            node_atts=torch.LongTensor([node_atts_padded[idx]]).to(device),
                            edges=torch.tensor(edges[gauss(num_nodes-2):gauss(num_nodes-1)]).unsqueeze(0).to(device)
                           )
                    data_full.append(data)
                data_list.append(tuple(data_full))
    return data_list



def DecLoader(data, batch_size, shuffle=True, device='cuda'):
    device = torch.device(device)
    if shuffle:
        shffle(data)
    while len(data) >= batch_size:
        batch_list = list()
        batch = data[:batch_size]
        for graph_batch in zip(*batch):
            for i in range(batch_size):
                graph_batch[i].to(device)
            loader = DataLoader(graph_batch, batch_size, False)
            batch_list.append(loader.__iter__().__next__().to(device))
        data = data[batch_size:]
        yield batch_list



        
        
        
def fetch_data(data_name):
    
    device = 'cuda'
    
    if data_name[-4:] == '.pth':
        return torch.load(data_name)
    
    
    f = h5py.File(data_name, 'r')
    graph_list = list()
    
    for graph in f.values():
        edge_list = graph['edgelist'].value
        node_attr = graph['node_attr'].value + 2
        val_acc = graph['val_acc'].value
        num_nodes = node_attr.size
        data = Data(
                    edge_index=torch.LongTensor(np.transpose(edge_list)).to(device),
                    num_nodes=num_nodes,
                    node_atts=torch.LongTensor(node_attr).to(device),
                    acc = torch.tensor([val_acc]).to(device)
            )
        graph_list.append(data)
        
    return graph_list

def data_to_longtensor(dataset):    
    data_list=[]
    for graph in dataset:
        data = Data(edge_index=torch.LongTensor(graph.edge_index),
    #             edit=torch.tensor([graph.edit]),
                node_atts=torch.LongTensor(graph.node_atts),
                num_nodes = graph.node_atts.size(0),
                acc = torch.tensor([graph.acc]),
#                 test_acc=torch.tensor([graph.test_acc]),
#                 training_time=torch.tensor([graph.training_time])
                   )  
        data_list.append(data)    
    return data_list


def my_round(data, dec=2):
    if isinstance(data, float):
        return np.round(data, dec)
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = np.round(value, dec)
    return data


        