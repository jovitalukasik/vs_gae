import os

import logging
logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import h5py
from time import time

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader

import mlflow as mlf

from models.GNNenc import NodeEmb, GraphEmb, GetAcc
from models.GNNprediction import GNNpred

from utils import sample_random, sample_edit, sample_even, fetch_data, my_round


import argparse
parser = argparse.ArgumentParser(description='doublin')
parser.add_argument('--method', type=str, help='randomly sampled NAS-Bench 101 Dataset', default='')
parser.add_argument('--data', type=str, help='training data', default='')
parser.add_argument('--training_size', type=int, help='size of training data to downsize from 1% to 100%', default='')
parser.add_argument('--epoch', type=str, help='training data', default=100)
args=parser.parse_args()



sampling_methods = {
    'random': (sample_random, 'data/training_70_dict.pth'),
}
method=args.method
path_results = "result/"+method
path_saved_acc = "save_acc/"+method



try:
    os.makedirs(path_results)
    print('Successfully created the directory %s ' % path_results)
    os.makedirs(path_saved_acc)
    print('Successfully created the directory %s ' % path_saved_acc)
except FileExistsError:
    print('Directory already exists %s ' % path_results)
    print('Directory already exists %s ' % path_saved_acc)


def main(inp, method,training_size, epoch):
    
    sample_func, data = sampling_methods[method]
    
    if data[-4:] == '.pth':
        data = torch.load(data)
    

    device = torch.device('cpu')
    batch_size = 128
    budget = epoch



    config = {
        'ndim': 250,
        'sdim': 56,
        'num_gnn_layers': 2,
        'g_aggr': 'gsum',
        'num_acc_layers': 4,
        'lr': 0.00001,
    }


    t0 = time()  
    test_dataset = fetch_data('data/test_data_20.pth')
    logging.info('Loaded test graphs in {} sec.'.format(round(time()-t0, 2)))
    t0 = time()
    val_dataset = fetch_data('data/validation_data_10.pth')
    logging.info('Loaded validation graphs model in {} sec.'.format(round(time()-t0, 2)))
    
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    criterion = nn.MSELoss()
    
    for num in [training_size]:
        ratio = np.round(num/100, 2) 
        run_name = '{}{}'.format(method, num)

        

        rmse_list = list()
        all_loss = list()
        best_rmse_list=[]
        for step in range(5):

            logger.info('sampling')
            sampled_dataset = sample_func(ratio, data)
            train_loader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=True)
            logger.info('start run {}_{} with {}% ({} graphs) '.format(run_name, step+1, num, len(sampled_dataset)))


            model = GNNpred(config['ndim'], config['sdim']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

            best_val = np.inf
            best_test = np.inf
            for epoch in range(int(budget)):
                loss = 0
                model.train()
                running_loss = torch.Tensor().to(device)
                for i, graph_batch in enumerate(train_loader):
                    graph_batch = graph_batch.to(device)
                    optimizer.zero_grad()
                    output = model(graph_batch.edge_index,
                                   graph_batch.node_atts,
                                   graph_batch.batch.to(device))
                    loss = criterion(output.view(-1), graph_batch.acc)
                    running_loss = torch.cat([running_loss, loss.view(-1)])
                    loss.backward()
                    optimizer.step()
                loss = torch.sqrt(torch.mean(running_loss)).item()
                all_loss.append(loss)

                logger.info('epoch {}:\tloss = {}'.format(epoch, my_round(loss, 4)))
            
                val_rmse, _ ,val_acc= evaluate(model, val_loader, device)

                logger.info('epoch {}:\tval_rmse = {}'.format(epoch, my_round(val_rmse, 4)))

                test_rmse, test_mae ,_= evaluate(model, test_loader, device)

                logger.info('epoch {}:\ttest_rmse = {}'.format(epoch, my_round(test_rmse, 4)))
                logger.info('epoch {}:\ttest_mae = {}'.format(epoch, my_round(test_mae, 4)))
                
                            
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_test = test_rmse
#                         save(model, run_name)
                rmse_list.append(best_val)
            best_rmse_list.append(best_val)
            logger.info('step {}:\tbest_test = {}'.format(step+1, my_round(best_test, 4)))
            
            torch.save(rmse_list, path_results+'/{}_all_rmse.pth'.format(run_name))
            logger.info('Saved all validation rmse to {}'.format(path_results))
                        
            torch.save(best_rmse_list, path_results+'/{}_best_rmse.pth'.format(run_name))
            logger.info('Saved best validation rmse of each run to {}'.format(path_results))
            
            torch.save(all_loss, path_results+'/{}_loss.pth'.format(run_name))
            logger.info('Saved trainings loss to {}'.format(path_results))
            
            torch.save(val_acc, path_saved_acc+'/{}_val_acc.pth'.format(run_name))
            logger.info('Saved true and predicted accuracy of validation set to {}'.format(path_saved_acc))
            
            _, _ ,train_acc= evaluate(model, train_loader, device)
            torch.save(train_acc, path_saved_acc+'/{}_train_acc.pth'.format(run_name))
            logger.info('Saved true and predicted accuracy of training set to {}'.format(path_saved_acc))
            
            
            logger.info('epoch {}:\ttest_rmse = {}'.format(epoch, my_round(test_rmse, 4)))


    return loss, val_rmse,test_rmse,test_mae, model.number_of_parameters()



def evaluate(model, data_loader, device):
    criterion = nn.MSELoss(reduction='none')
    model.eval()
    data_acc=torch.Tensor().to(device)
    pred_acc=torch.Tensor().to(device)
    test_loss = torch.Tensor().to(device)
    for graph_batch in data_loader:
        graph_batch.to(device)
        data_acc=torch.cat([data_acc, graph_batch.acc])
        with torch.no_grad():
            out = model(graph_batch.edge_index,
                        graph_batch.node_atts,
                        graph_batch.batch.to(device))  
        pred_acc=torch.cat([pred_acc, out.view(-1)])
        loss = criterion(out.view(-1), graph_batch.acc)
        test_loss = torch.cat([test_loss, loss])
    acc=torch.stack([data_acc,pred_acc])
    rmse = torch.sqrt(torch.mean(test_loss)).item()
    mae = torch.mean(torch.sqrt(test_loss)).item()
    return rmse, mae, acc







def save(model, name):
    torch.save(model.state_dict(), 'saved_models/{}/' + name)

   

    
if __name__ == '__main__':
    main(args.data, args.method, args.training_size, args.epoch)
    