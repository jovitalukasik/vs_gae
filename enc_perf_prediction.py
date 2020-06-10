import logging
logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
import pdb
import numpy as np
import time 
import json
import os
import sys
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader

from models import GNNpred

from utils import utils
from utils import sample_random, sample_edit, sample_even


import argparse
parser = argparse.ArgumentParser(description='GNN PerformancePrediciton')
parser.add_argument('--model', type=str, default='GNN-VSGAE')
parser.add_argument('--prediction_task', choices=['interpolation', 'extrapolation'], default='interpolation', help='predicition in which areas')
parser.add_argument('--save_interval', type=int, default=50, help='how many epochs to wait to save model')
parser.add_argument('--sampling', type=str, default='random', help='randomly (even/edit) sampled NAS-Bench 101 Dataset')
parser.add_argument('--train_data', type=str, help='training data in ../data', default='data/training_data_70.pth')
parser.add_argument('--validation_data', type=str, help='training data in ../data', default='data/validation_data_10.pth')
parser.add_argument('--test_data', type=str, help='training data in ../data', default='data/test_data_20.pth')
parser.add_argument('--training_size', type=int, help='size of training data to downsize from 1% to 100%', default='100')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int,  default=100)
parser.add_argument('--num_acc_layers', type=int, help='amount linear layer for regression', default=4)
parser.add_argument('--learning_rate', type=float, default=0.00001)
args=parser.parse_args()

args.save = 'experiments/performance_prediction/gnn/{}/{}/sampled-{}/pred-{}'.format(
    args.prediction_task,
    args.sampling,
    args.training_size,
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


sampling_methods = {
    'random': (sample_random),
    'edit': (sample_edit), 
    'even': (sample_even)
}



def main(args):
    
    method = args.sampling
    training_size = args.training_size
    sample_func= sampling_methods[method] 

    
    device = torch.device('cuda')
    batch_size = args.batch_size 
    budget = args.epochs
    
    logging.info("args = %s", args)

    with open('model_configs/gnn_config.json')  as json_file:
             config = json.load(json_file)


    config = {
        'num_gnn_layers': config['num_gnn_layers'],
        "dropout_prob": config['dropout_prob'],
        "gnn_hidden_dimensions":config['gnn_hidden_dimensions'],
        'gnn_node_dimensions': config['gnn_node_dimensions'],
        'g_aggr': 'gsum',
        'lr': args.learning_rate,
        'num_acc_layers': args.num_acc_layers, 
        'num_node_atts':5, 
        'dim_target':1,
        'batch_size':batch_size,
        'epochs': budget, 
        'training_size':training_size, 
        'sampling_method':method
    }
    
    with open(os.path.join(args.save, 'config.json'), 'w') as fp:
        json.dump(config, fp)
        
    logging.info("architecture configs = %s", config)
    
    criterion = nn.MSELoss()


    model = GNNpred(config['gnn_node_dimensions'], config['gnn_hidden_dimensions'], config['dim_target'],
                    config['num_gnn_layers'], config['num_acc_layers'], config['num_node_atts'], 
                            model_config=config).to(device)
       
    if args.prediction_task=='interpolation':
        #Load Test Data
        test_data=args.test_data
        t0 = time.time()  
        test_dataset = torch.load(test_data)
        test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

        logging.info('Loaded test graphs in {} sec.'.format(round(time.time()-t0, 2)))
    
  
    #Load Validation Data
    validation_data=args.validation_data
    t0 = time.time()

    val_dataset = torch.load(validation_data)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)      

       
    logging.info('Loaded validation graphs model in {} sec.'.format(round(time.time()-t0, 2)))
    ratio = np.round(training_size/100, 2) 
    
    
        
    logger.info('sampling')
    #Load Training data
    train_data=args.train_data
    sampled_dataset = sample_func(ratio, torch.load(train_data))
    train_loader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=True)

    logger.info('start training {} with {}% ({} graphs) '.format(args.model, training_size, len(sampled_dataset)))
    
    # Save Sampled Dataset
    filepath = os.path.join(args.save, 'sampled_dataset_{}.pth'.format(training_size))
    torch.save(sampled_dataset, filepath)


    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']) 
    
    for epoch in range(1, int(budget)+1):
        logging.info('epoch: %s', epoch)
       
        # training
        train_obj, train_results=train(train_loader, model, criterion, optimizer, config['lr'], epoch, device, 
                                       batch_size)
        logging.info('train metrics:  %s', train_results)

#       validation    
        valid_obj, valid_results = infer(val_loader, model, criterion, epoch, device, batch_size)
        logging.info('validation metrics:  %s', valid_results)
        
        if args.prediction_task=='interpolation':
    #       testing
            test_obj, test_results= test(test_loader, model, criterion, device, batch_size)
            logging.info('test metrics:  %s', test_results)
            config_dict = {
                'epochs': args.epochs,
                'loss': train_results["rmse"],
                'val_rmse': valid_results['rmse'],
                'test_rmse': test_results['rmse'],
                'test_mse': test_results['mse']
                }

        # Save the entire model
        if epoch % args.save_interval == 0:
            logger.info('save model checkpoint {}  '.format(epoch))
            filepath = os.path.join(args.save, 'model_{}.obj'.format(epoch))
            torch.save(model.state_dict(), filepath)
        

        with open(os.path.join(args.save, 'results.txt'), 'w') as file:
                json.dump(str(config_dict), file)
                file.write('\n')




def train(train_loader,model, criterion, optimizer, lr, epoch, device, batch_size):
    objs = utils.AvgrageMeter()
    
    # TRAINING
    preds = []
    targets = []
        
    model.train()

    for step, graph_batch in enumerate(train_loader):
        graph_batch = graph_batch.to(device)
        pred = model(graph_batch=graph_batch).view(-1)
        loss = criterion(pred, (graph_batch.acc))

        preds.extend((pred.detach().cpu().numpy()))
        targets.extend(graph_batch.acc.detach().cpu().numpy())
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = graph_batch.num_graphs
        objs.update(loss.data.item(), n)
    
#     logging.info('train %03d %.5f', step, objs.avg)

    train_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
    
    return objs.avg, train_results


def infer(val_loader, model, criterion, epoch, device, batch_size):
    objs = utils.AvgrageMeter()

    # VALIDATION
    preds = []
    targets = []

    model.eval()
        
    for step, graph_batch in enumerate(val_loader):
        graph_batch = graph_batch.to(device)
        pred = model(graph_batch=graph_batch).view(-1)
        loss = criterion(pred, (graph_batch.acc))

        preds.extend((pred.detach().cpu().numpy()))
        targets.extend(graph_batch.acc.detach().cpu().numpy())
        n = graph_batch.num_graphs
        objs.update(loss.data.item(), n)

#     logging.info('valid %03d %.5f', step, objs.avg)

    val_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)

    return objs.avg, val_results


def test(test_loader, model, criterion, device, batch_size):
    objs = utils.AvgrageMeter()
    preds = []
    targets = []
    
    model.eval()
     
    for step, graph_batch in enumerate(test_loader):
        graph_batch = graph_batch.to(device)
        pred = model(graph_batch=graph_batch).view(-1)
        loss = criterion(pred, (graph_batch.acc))

        preds.extend((pred.detach().cpu().numpy()))
        targets.extend(graph_batch.acc.detach().cpu().numpy())
        n = graph_batch.num_graphs
        objs.update(loss.data.item(), n)

    test_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
#     logging.info('test metrics %s', test_results)

    return objs.avg, test_results

 

    
if __name__ == '__main__':
    main(args)
    
