"""
Code for testing the autoencoder from official repository of 
"D-VAE: A Variational Autoencoder for Directed Acyclic Graphs", Advances in Neural Information Processing Systems 2019
https://github.com/muhanzhang/D-VAE
"""



import logging
logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
import pdb
import numpy as np
import networkx as nx
import h5py
import time 
import json
import operator
import json
import pickle
import scipy
import glob
from tqdm import tqdm
from tqdm import tqdm_notebook
import os
import sys
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import GNNVAE

from utils import utils
from utils import DecLoader, prep_data
from utils import data_to_longtensor
from utils import prep_data, batch2graph

import argparse
parser = argparse.ArgumentParser(description=' GNN Graphautoencoder-training')
parser.add_argument('--model', type=str, default='GNN-VSGAE')
parser.add_argument('--train_data', type=str, help='training data in ../data', default='data/training_data_90.pth')
parser.add_argument('--test_data', type=str, help='test data in ../data for VAE ability checks', default='data/validation_data_10.pth')
parser.add_argument('--save_interval', type=int, default=100, help='how many epochs to wait to save model')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int,  default=300)
parser.add_argument('--num_gnn_layers', type=int, help='amount of propagation steps in GNN', default=2)
parser.add_argument('--gnn_hidden_dimensions', type=int, help='graph embedding dimension', default=56)
parser.add_argument('--gnn_node_dimensions', type=int, help='graph node embedding dimension', default=250)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--dropout_prob', type=float,  default=0.0)
parser.add_argument('--beta', type=float,  default=5e-3)
parser.add_argument('--comments', type=str, default='')
parser.add_argument('--default', action='store_true', default=False, help='if True, use values from args')
parser.add_argument('--test', action='store_true', default=False, help='Testing the VAE on Autoencoding Ability with test data')

args=parser.parse_args()

args.save = 'experiments/VAE/vae-{}'.format(
    time.strftime(
        "%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main(args):

    device = torch.device('cuda')
    
    logging.info("args = %s", args)
    #Load Model Configs
    if not args.default:
        with open('model_configs/gnn_config.json')  as json_file:
            config = json.load(json_file)

        config = {
            'epochs':args.epochs,
            'num_gnn_layers': config['num_gnn_layers'],
            'learning_rate': config['learning_rate'],
            "dropout_prob": config['dropout_prob'],
            "gnn_hidden_dimensions":config['gnn_hidden_dimensions'],
            'gnn_node_dimensions': config['gnn_node_dimensions'],
            'g_aggr': 'gsum',
            'beta':config['beta'] ,
            'num_node_atts':5,
            'batch_size': config['batch_size'], 
            }

        batch_size=config['batch_size']

    else:
        config = {
            'num_gnn_layers': args.num_gnn_layers,
            'learning_rate': args.learning_rate,
            "dropout_prob": args.dropout_prob,
            'learning_rate': args.learning_rate,
            "gnn_hidden_dimensions": args.gnn_hidden_dimensions,
            'gnn_node_dimensions': args.gnn_node_dimensions,
            'g_aggr': 'gsum',
            'beta': args.beta, 
            'num_node_atts':5, 
            'batch_size':args.batch_size, 
            'epochs':args.epochs,
            
            }
        batch_size=args.batch_size
    budget = args.epochs
    
    logging.info("true architecture configs = %s", config)
        
    with open(os.path.join(args.save, 'config.json'), 'w') as fp:
        json.dump(config, fp)
    
    
    criterion = nn.MSELoss()
    
    #Load Models

    model = GNNVAE(config['gnn_node_dimensions'], config['gnn_hidden_dimensions'], config['num_gnn_layers'], 
                        config['num_node_atts'], beta=config['beta'], model_config=config).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) 
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)   

    #Load Training data
    logging.info('Prep Train Dataset {}'. format(args.train_data))
    train_data=args.train_data
    max_num_nodes = 7
    train_set = prep_data(train_data, max_num_nodes=max_num_nodes, training=True)  

    
    logger.info('start training {}'. format(args.model))

    for epoch in range(1, int(budget)+1):
        logging.info('epoch: %s', epoch)
       
        # training
        train_obj=train(train_set, model, optimizer, config['learning_rate'], epoch, device, batch_size)
        scheduler.step(train_obj)
            
            
        # Save the model
        if epoch % args.save_interval == 0:
            logger.info('save model checkpoint {}  '.format(epoch))
            filepath = os.path.join(args.save, 'model_{}.obj'.format(epoch))
            torch.save(model.state_dict(), filepath)


        config_dict = {
                'epochs': epoch,
                'loss': train_obj,
                }

        
        with open(os.path.join(args.save, 'results.txt'), 'a') as file:
                json.dump(str(config_dict), file)
                file.write('\n')
                
    if args.test:
        state_dict = model.state_dict()

        model.load_state_dict(state_dict)
        
        #Load Test Data
        data_list_test = prep_data(args.test_data)
        test_set=data_to_longtensor(data_list_test)
        
        #Load Train Data in different format than for training
        train_data=args.train_data
        data_list_train = prep_data(train_data)
        train_set=data_to_longtensor(data_list_train)
        
        
        #Reconstruction Accuracy
        logger.info('Run: Test Dataset for Reconstruction Accuracy {}'.format(args.test_data))
        rec_acc= recon_accuracy(test_set,model, state_dict, device)
        logger.info('Reconstruction Accuracy on test set for model {} is {}'.format(args.model,rec_acc))
        
        #Prior Ability, Uniqueness, Novelty
        logger.info('Run: Train Dataset for Validity Tests {}'.format(args.train_data))

        logger.info('Extract mean and std of latent space ')
        save_latent_representations(epoch, train_set, test_set, model ,state_dict, 1, device, data_name='nas101')


        batch_size=2048
        Z_train,  V_train  = extract_latent_true(train_set, model, state_dict, batch_size, device)

        n_latent_points=1000
        prior, unique, novel= prior_validity(train_set,test_set,model, state_dict, Z_train,n_latent_points,
                                             device, scale_to_train_range=True)

        logger.info('Prior Validity on train set for model {} is {}'.format(args.model,prior))
        logger.info('Unique Graphs from train set  for model {} is {}'.format(args.model,unique))
        logger.info('Novel Graphs from train set set for model {} is {}'.format(args.model,novel))

        config_dict = {
            'rec_acc':rec_acc,
            'prior': prior,
            'unique':unique,
            'novel':novel
               }



        with open(os.path.join(args.save, 'results_validity.txt'), 'w') as file:
            json.dump(str(config_dict), file)
            file.write('\n')            


def train(train_loader,model, optimizer, lr, epoch, device, batch_size):
    objs = utils.AvgrageMeter()
    
    # TRAINING
        
    model.train()
      
    for step,graph_batch in enumerate(DecLoader(train_loader, batch_size, shuffle=True, device=device)):
        loss = model(graph_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = graph_batch[0].num_graphs
        objs.update(loss.data.item(), n)

        
    logging.info('train %03d %.5f', step, objs.avg)
    
    return objs.avg


def parse_graph_to_nx(e, label):
    G=nx.DiGraph()
    for i in range(e.shape[1]):
          G.add_edge(e[0][i].item(),e[1][i].item())
    for i in range(len(G)):
        G.nodes[i]['Label']= label[i].item()
    return G



def is_same_DAG(g0, g1):
    attr0=(nx.get_node_attributes(g0, 'Label'))
    attr1=(nx.get_node_attributes(g1, 'Label'))
    # note that it does not check isomorphism
    if g0.__len__() != g1.__len__():
        return False
    for vi in range(g0.__len__()):
        if attr0[vi] != attr1[vi]:
            return False
        if set(g0.pred[vi]) != set(g1.pred[vi]):
            return False
    return True


def ratio_same_DAG(G0, G1):
    # how many G1 are in G0
    res = 0
    for g1 in tqdm(G1):
        for g0 in G0:
            if is_same_DAG(g1, g0):
                res += 1
                break
    return res / len(G1)


def is_valid_DAG(g, START_TYPE=0, END_TYPE=1):
    # Check if the given igraph g is a valid DAG computation graph
    # first need to have no directed cycles
    # second need to have no zero-indegree nodes except input
    # third need to have no zero-outdegree nodes except output
    # i.e., ensure nodes are connected
    # fourth need to have exactly one input node
    # finally need to have exactly one output node
    attr=(nx.get_node_attributes(g, 'Label'))
    res = nx.is_directed_acyclic_graph(g)
    n_start, n_end = 0, 0
    for vi in range(g.__len__()):
        if attr[vi] == START_TYPE:
            n_start += 1
        elif attr[vi] == END_TYPE:
            n_end += 1
        if g.in_degree(vi) == 0 and attr[vi] != START_TYPE:
            return False
        if g.out_degree(vi) == 0 and attr[vi] != END_TYPE:
            return False
    return res and n_start == 1 and n_end == 1


# Test Reconstruction Accuracy

def recon_accuracy(test_set,model, state_dict, device):
    data_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model.load_state_dict(state_dict)
    model.eval()

    encode_times = 10 # sample embedding 10 times for each Graph
    decode_times = 10 # decode each embedding 10 times
    n_perfect = 0
    pbar = tqdm(data_loader) #10% of Dataset
    for i, graph in enumerate(pbar):
        g=parse_graph_to_nx(graph.edge_index, graph.node_atts)
        graph=graph.to(device)
        _, _, _, mean, log_var, _ = model.inference(graph, sample=True)
        for _ in range(encode_times):
            _,_,_,_,_, z =  model.inference(mean, sample=True, log_var=log_var)
            for _ in range(decode_times):
                _, label, edges, _, _, _ = model.inference(z)
                try:
                    g_recon=parse_graph_to_nx(edges, label[0])
                    n_perfect += (int(is_same_DAG(g, g_recon)))
                except:
                    continue
                    
    acc = n_perfect / (len(test_set) * encode_times * decode_times)
    print('Recon accuracy from Test Set: {:.5f}'.format(acc))
    return acc




def extract_latent(train_data, model, state_dict, infer_batch_size, device):
    print('Scaling to Training Data Range')
    data_loader = DataLoader(train_data, batch_size=infer_batch_size, shuffle=False)
    model.load_state_dict(state_dict)
    model.eval()
    Z = []
    Y=[]
    g_batch = []
    pbar = tqdm(data_loader) 
    for i, graph in enumerate(pbar):
        graph.to(device)
        _, _, _, mean, _, _ = model.inference(graph, sample=True) 
        mean = mean.cpu().detach().numpy()
        Z.append(mean)
        Y.append(graph.acc.cpu()) 
    return np.concatenate(Z, 0), torch.cat(Y,0).numpy()


def save_latent_representations(epoch, train_data, test_data, model ,state_dict, infer_batch_size, device, data_name):
    Z_train, Y_train = extract_latent(train_data, model, state_dict, infer_batch_size, device)
    Z_test, Y_test = extract_latent(test_data, model, state_dict, infer_batch_size, device)
    latent_pkl_name = os.path.join(args.save,  data_name+
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.save, data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)
    scipy.io.savemat(latent_mat_name, 
                     mdict={
                         'Z_train': Z_train, 
                         'Z_test': Z_test, 
                         'Y_train': Y_train, 
                         'Y_test': Y_test
                         }
                     )




def extract_latent_true(train_data, model, state_dict, infer_batch_size, device):
    data_loader = DataLoader(train_data, batch_size=infer_batch_size, shuffle=False)
    model.load_state_dict(state_dict)
    model.eval()
    Z = []
    V=[]
    g_batch = []
    pbar = tqdm(data_loader) 
    for i, graph in enumerate(pbar):
        graph.to(device)
        _, _, _, mean, log_var, _ = model.inference(graph, sample=True) 
        mean = mean.cpu().detach().numpy()
        log_var = log_var.cpu().detach().numpy()
        Z.append(mean)
        V.append(log_var) 
    return np.concatenate(Z, 0), np.concatenate(V, 0)



def prior_validity(train_data,test_data, model, state_dict , Z_train , n_latent_points, device ,scale_to_train_range=False):
    data_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    data_loader_test = DataLoader(test_data, batch_size=1, shuffle=False)
    model.load_state_dict(state_dict)
    model.eval()
    if scale_to_train_range:
        z_mean, z_std = Z_train.mean(0), Z_train.std(0)
        z_mean, z_std = torch.FloatTensor(z_mean).to(device), torch.FloatTensor(z_std).to(device)
    nz=z_mean.size(0)
    false_decoding=[]
    false_sample=[]
    decode_times = 10
    n_valid = 0
    amount=0
    print('Prior validity experiment begins...')
    G = []
    G_valid = []
    G_valid_str=[]
    pbar = tqdm(range(n_latent_points))
    for i in pbar:
        z = torch.randn(1, nz).to(device)
        z = z * z_std + z_mean  # move to train's latent range
        for j in range(decode_times):
            try:
                g_str, label, edges, _, _, _ = model.inference(z)
                g_str_batch=batch2graph(g_str)
                for graph in g_str_batch:
                    g=parse_graph_to_nx(graph[1], graph[0])
                    G.extend(g)
                    amount+=1
                    if is_valid_DAG(g, START_TYPE=1, END_TYPE=0):
                        n_valid += 1
                        G_valid_str.append(g_str)
                        G_valid.append(g)
                else: 
                    false_sample.append(z)
                    false_decoding.append(graph)
            except:
                continue

    r_valid = n_valid / (n_latent_points * decode_times)
    print('Ratio of valid decodings from the prior: {:.4f}'.format(r_valid))
    print('amount /n:',amount)
    
    
    G_str = [str(g[0].cpu().numpy()) for g in G_valid_str]
    r_unique = len(set(G_str)) / len(G_str) if len(G_str)!=0 else 0.0
    print('Ratio of unique decodings from the prior: {:.4f}'.format(r_unique))
    
    G_train=[]
    for graph in data_loader:
        g=parse_graph_to_nx(graph.edge_index, graph.node_atts)
        G_train.append(g)

    r_novel = 1 - ratio_same_DAG(G_train, G_valid)
    print('Ratio of novel graphs out of training data: {:.4f}'.format(r_novel))
    
    return r_valid, r_unique, r_novel


    
if __name__ == '__main__':
    main(args)
    
