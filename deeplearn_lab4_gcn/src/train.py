from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score

from utils import load_data, accuracy, load_ppi_data
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
'''
self-loop, pairnorm. dropedge, layer_num, activate
'''
parser.add_argument('--drop_edge', type=float, default=0.,
                    help='DropEdge rate (1 - keep probability).')
parser.add_argument('--pair_norm', type=bool, default=False,
                    help='Wether to use PairNorm or not')
parser.add_argument('--self_loop', type=bool, default=False,
                    help='Whether to use Self-Loop or not')
parser.add_argument('--layer_num', type=int, default=2,
                    help='How many GC-layers are going to be used')
parser.add_argument('--activate', type=str, default='relu',
                    help='Which kind of activation function is going to be used')
parser.add_argument('--dataset', type=str, default='citeseer',
                    help='Select which dataset to conduct experiment')
parser.add_argument('--task', type=str, default='nodecls',
                    help='nodecls(Node classification) or linkpred(Link Prediction)')


args = parser.parse_args()
print(args.self_loop)
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.task == 'nodecls':
    if args.dataset == 'cora' or args.dataset == 'citeseer':
        adj, features, labels, idx_train, idx_val, idx_test = load_data(
                                                            dataset=args.dataset, 
                                                            task=args.task,
                                                         self_loop=args.self_loop)
        # Model and optimizer
        model = GCN(in_channels=features.shape[1],
                    hid_channels=args.hidden,
                    out_channels=labels.max().item() + 1,
                    dropout=args.dropout,
                    layer_num=args.layer_num,
                    activation=args.activate,
                    drop_edge=args.drop_edge,
                    pair_norm=args.pair_norm)
        
    elif args.dataset == 'ppi':
        adj, features, labels, idx_train, idx_val, idx_test = load_ppi_data(
                                                            task=args.task,
                                                            self_loop=args.self_loop)
        # Model and optimizer
        model = GCN(in_channels=features.shape[1],
                    hid_channels=args.hidden,
                    out_channels=args.hidden,
                    dropout=args.dropout,
                    layer_num=args.layer_num,
                    activation=args.activate,
                    drop_edge=args.drop_edge,
                    pair_norm=args.pair_norm)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    
elif args.task == 'linkpred':
    if args.dataset == 'cora' or args.dataset == 'citeseer':
        adj, features, train_edges, val_edges, test_edges, \
                        train_label, val_label, test_label = load_data(
                                                            dataset=args.dataset, 
                                                            task=args.task,
                                                            self_loop=args.self_loop) 
    elif args.dataset == 'ppi':
        adj, features, train_edges, val_edges, test_edges, \
                        train_label, val_label, test_label = load_ppi_data(
                                                            task=args.task,
                                                            self_loop=args.self_loop) 
    '''
    train_edges = list [[src_pos_1,...,src_pos_m, src_neg_1,...,src_neg_m],
                        [dst_pos_1,...,dst_pos_m, dst_neg_1,...,dst_neg_m]]
    train_label = torch.tensor([1, 1, 1,...,1, 0, ..., 0], dtype=long)
    '''
    # Model and optimizer
    model = GCN(in_channels=features.shape[1],
                hid_channels=args.hidden,
                out_channels=args.hidden,
                dropout=args.dropout,
                layer_num=args.layer_num,
                activation=args.activate,
                drop_edge=args.drop_edge,
                pair_norm=args.pair_norm)
    
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        train_label = train_label.cuda()
        val_label = val_label.cuda()
        test_label = test_label.cuda()

else:
    raise Exception('task({}) is supposed to belong to \{"nodecls", "linkpred"\}.'.format(task))

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.task == 'nodecls':
    if args.dataset != 'ppi':
        criterion = F.nll_loss
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
elif args.task == 'linkpred':
    criterion = torch.nn.BCEWithLogitsLoss()

val_performances = list()
test_performances = list()

def train(epoch, task='nodecls'):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    if task == 'nodecls':
        if args.dataset != 'ppi':
            output = model(features, adj)
            loss_train = criterion(output[idx_train], labels[idx_train])
        else:
            output = model(x=features, adj=adj, ppi=True)
            loss_train = criterion(output[idx_train], labels[idx_train].float())

        if args.dataset != 'ppi':
            acc_train = accuracy(output[idx_train], labels[idx_train])
        else:
            preds = (output[idx_train] > 0).float().cpu()
            #print(labels[idx_train].shape, preds.shape)
            f1_train = f1_score(labels[idx_train].cpu(), preds, average='micro')

    elif task == 'linkpred':
        output = model(features, adj, 'linkpred', train_edges)
        loss_train = criterion(output, train_label)
        logits = torch.sigmoid(output)
        auc_train = roc_auc_score(train_label.cpu().numpy(), logits.detach().cpu().numpy())

    loss_train.backward()
    optimizer.step()

    
    model.eval()
    if task == 'nodecls':
        if args.dataset != 'ppi':
            output = model(features, adj)
            loss_val = criterion(output[idx_val], labels[idx_val])
        else:
            output = model(x=features, adj=adj, ppi=True)
            loss_val = criterion(output[idx_val], labels[idx_val].float())

        if args.dataset != 'ppi':
            acc_val = accuracy(output[idx_val], labels[idx_val])
        else:
            preds = (output[idx_val] > 0).float().cpu()
            f1_val = f1_score(labels[idx_val].cpu(), preds, average='micro')


        
        if args.dataset != 'ppi':
            loss_test = criterion(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
        else:
            loss_test = criterion(output[idx_test], labels[idx_test].float())
            preds = (output[idx_test] > 0).float().cpu()
            f1_test = f1_score(labels[idx_test].cpu(), preds, average='micro')

        if args.dataset != 'ppi':
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'loss_val: {:.4f}'.format(loss_test.item()),
                'acc_val: {:.4f}'.format(acc_test.item()),
                'time: {:.4f}s'.format(time.time() - t))
            val_performances.append(acc_val.item())
            test_performances.append(acc_test.item())
        else:
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'f1_train: {:.4f}'.format(f1_train),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'f1_val: {:.4f}'.format(f1_val),
                'loss_val: {:.4f}'.format(loss_test.item()),
                'f1_test: {:.4f}'.format(f1_test),
                'time: {:.4f}s'.format(time.time() - t))
            val_performances.append(f1_val.item())
            test_performances.append(f1_test.item())

    elif task == 'linkpred':
        output = model(features, adj, 'linkpred', val_edges)
        loss_val = criterion(output, val_label)
        logits = torch.sigmoid(output)
        auc_val = roc_auc_score(val_label.cpu().numpy(), logits.detach().cpu().numpy())

        output = model(features, adj, 'linkpred', test_edges)
        loss_test = criterion(output, test_label)
        logits = torch.sigmoid(output)
        auc_test = roc_auc_score(test_label.cpu().numpy(), logits.detach().cpu().numpy())

        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'auc_train: {:.4f}'.format(auc_train),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'auc_val: {:.4f}'.format(auc_val),
            'loss_test: {:.4f}'.format(loss_test.item()),
            'auc_test: {:.4f}'.format(auc_test),
            'time: {:.4f}s'.format(time.time() - t))

        val_performances.append(auc_val)
        test_performances.append(auc_test)

    

def test(task='nodecls'):
    if task == 'nodecls':
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))
    elif task == 'linkpred':
        model.eval()
        with torch.no_grad():
            output = model(features, adj, 'linkpred', test_edges)
            loss_test = criterion(output, test_label)
            logits = torch.sigmoid(output)
        auc_test = roc_auc_score(test_label.cpu().numpy(), logits.detach().cpu().numpy())
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "auc score= {:.4f}".format(auc_test))

def output_best(val_performances, test_performances, task='nodecls'):
    val_performances = np.array(val_performances)
    max_id = np.argmax(val_performances)
    if task == 'linkpred':
        print("Test set results (with best validation performance):",
            "auc score= {:.4f}".format(test_performances[max_id]))
    else:
        if args.dataset != 'ppi':
            print("Test set results (with best validation performance):",
                "acc = {:.4f}".format(test_performances[max_id]))
        else:
            print("Test set results (with best validation performance):",
                "f1_score = {:.4f}".format(test_performances[max_id]))




# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch, args.task)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
#test(args.task)

output_best(val_performances, test_performances)