import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import negative_sampling

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset, task, self_loop):
    
    if dataset == 'cora':
        path = "../datasets/cora/"
        dataset = "cora"
    elif dataset == 'citeseer':
        #path = "../datasets/citeseer_new/"
        path = "/data2/home/zhaoyi/labs/USTC-labs/deeplearn_lab4_gcn/datasets/citeseer_new/"
        dataset = "citeseer"
    
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    np.random.shuffle(idx_features_labels)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    
    temp1 = map(idx_map.get, edges_unordered.flatten())
    temp2 = list(temp1)
    x = list(edges_unordered.flatten())
    print(x[462])
    for i in range(len(temp2)):
        elem = temp2[i]
        try:
            elem = int(elem)
        except TypeError:
            print(i)

    edges = np.array(temp2, dtype=np.int32).reshape(edges_unordered.shape)
    '''
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    '''
    print('You are currently running {} task on {} dataset...'.format(task, dataset))
    if task == 'linkpred':
        edge_num = edges.shape[0]
        shuffled_ids = np.random.permutation(edge_num)
        test_set_size = int(edge_num * 0.15)
        val_set_size = int(edge_num * 0.15)
        test_ids = shuffled_ids[ : test_set_size]
        val_ids = shuffled_ids[test_set_size : test_set_size + val_set_size]
        train_ids = shuffled_ids[test_set_size + val_set_size : ]

        train_pos_edges = torch.tensor(edges[train_ids], dtype=int)
        val_pos_edges = torch.tensor(edges[val_ids], dtype=int)
        test_pos_edges = torch.tensor(edges[test_ids], dtype=int)

        train_pos_edges = torch.transpose(train_pos_edges, 1, 0)
        # shape = [2, train_pos_edge_num]
        val_pos_edges = torch.transpose(val_pos_edges, 1, 0)
        test_pos_edges = torch.transpose(test_pos_edges, 1, 0)

        def negative_sample(pos_edges, nodes_num):
            '''
            pos_edges = [[src_1,...],
                        [dst_1,...]]
            '''
            neg_edges = negative_sampling(
                edge_index=pos_edges,
                num_nodes=nodes_num,
                num_neg_samples=pos_edges.shape[1],
                method='sparse'
            )
            edges = torch.cat((pos_edges, neg_edges), dim=-1)
            '''
            edges = [[src_1,src_2,...,src_m],
                    [dst_1,dst_2,...,dst_m]]
            shape = [2, 2*train_edge_num]
            '''
            edges_label = torch.cat((
                torch.ones(pos_edges.shape[1]),
                torch.zeros(neg_edges.shape[1])
            ),dim=0)
            # size = [2*train_edge_num]
            return edges, edges_label
        
        train_edges, train_label = negative_sample(train_pos_edges, idx.shape[0])
        val_edges, val_label = negative_sample(val_pos_edges, idx.shape[0])
        test_edges, test_label = negative_sample(test_pos_edges, idx.shape[0])
        
        adj = sp.coo_matrix((np.ones(train_pos_edges.shape[1]), (train_pos_edges[0], train_pos_edges[1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
    
        if self_loop == True:
            adj = normalize(adj + sp.eye(adj.shape[0]))
        else:
            adj = normalize(adj)

        features = torch.FloatTensor(np.array(features.todense()))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        train_edges = train_edges.tolist()
        val_edges = val_edges.tolist()
        test_edges = test_edges.tolist()
        train_label = train_label.type(torch.float)
        val_label = val_label.type(torch.float)
        test_label = test_label.type(torch.float)

        return adj, features, train_edges, val_edges, test_edges, \
                    train_label, val_label, test_label

    elif task == 'nodecls':
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
    
        if self_loop == True:
            adj = normalize(adj + sp.eye(adj.shape[0]))
        else:
            adj = normalize(adj)
        
        # split train || val || test
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
        
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test
    
    else:
        raise Exception("hyper-parameter `task` belongs to \{'nodecls', 'linkpred'\}.")


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):

    preds = output.max(1)[1].type_as(labels)
        
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
