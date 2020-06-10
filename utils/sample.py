import h5py
import torch
import numpy as np
import numpy.random as rnd
from numpy.random import shuffle
from collections import defaultdict
from torch_geometric.data import Data, DataLoader


def sample_random(ratio, graphs): 
    graph_dict = defaultdict(set)
    for graph in graphs:
        try:
            n=graph.node_atts.numpy().size
        except:    
            n=graph[0].node_atts.numpy().size
        graph_dict[n].add(graph) 
    sampled_list = list()
    for graph_list in graph_dict.values():
        graph_list = list(graph_list)
        shuffle(graph_list)
        length = len(graph_list)
        cut = int(np.round(length*ratio))
        sampled_list += graph_list[:cut]
    return sampled_list


def graphs_to_dict(graphs):
    trainings_dict = defaultdict(set)
    for graph in graphs:
            try:
                e=graph.edit.numpy().item()
            except:  
                e=graph[0].edit.numpy().item()
            trainings_dict[e].add(graph)        
    return trainings_dict



def training_edit_subsampling(share, trainings_dict):
    sample_share=share 
    subsampling_set=defaultdict(set)
    for edit, graph in trainings_dict.items(): 
        N=len(graph)
        s=0
        for gr in graph:
            subsampling_set[edit].add(gr)
            s+=1
            if s>=sample_share*N:
                break
    return subsampling_set

def sample_edit(ratio, graph_data):
    data_dict = graphs_to_dict(graph_data)
    sampled_dict = training_edit_subsampling(ratio, data_dict)
    graph_list = list()
    for graphs in sampled_dict.values():
        for graph in graphs:
            graph_list.append(graph)
    return graph_list



def sort_in_bins(X_sorted_dim, X_min, X_max, num_buckets, random=True, dim=4):
    num_nodes = X_sorted_dim[0].shape[0]
    X_buckets = {n: [] for n in range(num_nodes)}
    centers = [list() for _ in range(dim)]
    centers

    dist = np.abs(X_max-X_min)
    step = dist / num_buckets

    for d in range(dim):
        X_sorted = X_sorted_dim[d]
        eps = rnd.random() / num_buckets
        lim = X_min + eps
        center = lim - step/2
        centers[d].append(center)
        bucket = 0
        for row in X_sorted:
            idx = int(np.round(row[0]))
            if row[d+1] <= lim:            
                X_buckets[idx] += [bucket]
            else:
                lim += step
                center += step
                centers[d].append(center)
                bucket += 1
                X_buckets[idx] += [bucket]
                
    return X_buckets, centers


def get_bucket_dict(X_buckets):
    bucket_dict = dict()
    for idx, bucket in X_buckets.items():
        bucket_dict[tuple(bucket)] = bucket_dict.get(tuple(bucket), []) + [idx]
    return bucket_dict


def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def sample_even(ratio, graphs):
    r = int(np.round(100*ratio))

    needed_buckets = torch.load('data/even_sampling/needed_buckets.pth')
    num_buckets = needed_buckets[r]

    X_sorted_dim, X_min, X_max = torch.load('data/even_sampling/X_sorted_dim.pth')
    X_buckets, centers = sort_in_bins(X_sorted_dim, X_min, X_max, num_buckets=num_buckets)
    bucket_dict = get_bucket_dict(X_buckets)

    X = torch.load('data/even_sampling/training_70_d4.pth')


    num_nodes = X.shape[0]
    size = int(np.round(ratio*num_nodes))


    sampled_idx = list()
    for bucket, idx_list in bucket_dict.items():
        center = np.array([centers[d][i] for d, i in enumerate(bucket)])
        nodes = torch.index_select(torch.from_numpy(X), 0, torch.tensor(idx_list)).numpy() 
        idx_closest = idx_list[closest_node(center, nodes)] 
        sampled_idx.append(idx_closest)

        idx_list.remove(idx_closest)
        bucket_dict[bucket] = idx_list


    while len(sampled_idx) > size:
        sampled_idx.pop(rnd.choice(range(len(sampled_idx))))

    while len(sampled_idx) < size: 
        missing = size - len(sampled_idx)
        bucket_list = list(bucket_dict.keys())
        if missing < len(sampled_idx):
            rnd.shuffle(bucket_list)
            bucket_list = bucket_list[:missing]
        for bucket in bucket_list:
            idx_list = bucket_dict[bucket]
            if len(idx_list) == 0:
                continue       
            idx = rnd.choice(idx_list)
            sampled_idx.append(idx)
            idx_list.remove(idx)
            bucket_dict[bucket] = idx_list
            
    
    if isinstance(graphs, torch.Tensor):
        return torch.index_select(graphs, 0, torch.tensor(sampled_idx))
    
    else:
        return [graphs[idx] for idx in sampled_idx]
