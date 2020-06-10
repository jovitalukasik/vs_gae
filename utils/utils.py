import numpy as np
from scipy.stats import norm, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score
import os
import shutil

import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import DataLoader as BatchLoader

import igraph
import pygraphviz as pgv
from PIL import Image


def evaluate_metrics(y_true, y_pred, prediction_is_first_arg):
    """
    Create a dict with all evaluation metrics
    """

    if prediction_is_first_arg:
        y_true, y_pred = y_pred, y_true

    metrics_dict = dict()
    metrics_dict["mse"] = np.round(mean_squared_error(y_true, y_pred),4)
    metrics_dict["rmse"] = np.round(np.sqrt(metrics_dict["mse"]),4)
#     metrics_dict["r2"] = r2_score(y_true, y_pred)
#     metrics_dict["kendall_tau"], p_val = kendalltau(y_true, y_pred)
#     metrics_dict["spearmanr"] = spearmanr(y_true, y_pred).correlation

    return metrics_dict

def encoder_evaluation(data_set, Encoder,device, batch_size,  shuffle, log_var=False):
    enc_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False) 
    Encoder.eval()
    if log_var:
        with torch.no_grad():
            var=torch.Tensor().to(device)
            acc = torch.Tensor().to(device)
            for idx, batch in enumerate(enc_loader):
                batch.to(device)
                _, log_var = Encoder(batch.edge_index,
                                      batch.node_atts,
                                      batch.batch)
                var = torch.cat([var, log_var])
                acc = torch.cat([acc, batch.acc], 0)          
        var_and_acc = torch.cat([var, acc.unsqueeze(1)], 1)
        data_loader = BatchLoader(var_and_acc, batch_size=batch_size, shuffle=shuffle) 
        
    else:    
        with torch.no_grad():
            x = torch.Tensor().to(device)
            acc = torch.Tensor().to(device)
            for idx, batch in enumerate(enc_loader):
                batch.to(device)
                mean, _ = Encoder(batch.edge_index,
                                      batch.node_atts,
                                      batch.batch)
                x = torch.cat([x, mean])
                acc = torch.cat([acc, batch.acc], 0)          
        x_and_acc = torch.cat([x, acc.unsqueeze(1)], 1)
        data_loader = BatchLoader(x_and_acc, batch_size=batch_size, shuffle=shuffle) 

    return data_loader
    
    


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        
        
        
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
            
            
