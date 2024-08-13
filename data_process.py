import pickle
import scipy.io
import numpy as np
import scipy.sparse
import torch
import csv
import json
from os import path
from torch_geometric.data import Dataset, Data
import os
from sklearn.preprocessing import label_binarize
import random


def load_data(args):
    dataset_name = args.dataset
    data_dir = args.data_dir
    dataset_name = "ogbn_arxiv" if dataset_name == "ogbn-arxiv" else dataset_name
    
    if os.path.exists(path.join(data_dir, f"{dataset_name}/{dataset_name}.pkl")):
        with open (path.join(data_dir, f"{dataset_name}/{dataset_name}.pkl"), 'rb') as f:
            return pickle.load(f)
    
    if dataset_name == "twitch":
        graphs_list = ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW']
        datalist = []
        for graph_name in graphs_list:
            data = load_twitch(data_dir, graph_name)
            datalist.append(data)

        with open (path.join(data_dir, f"{dataset_name}/{dataset_name}.pkl"), 'wb') as f:
            pickle.dump([datalist, graphs_list], f) 

    elif dataset_name == "fb100":
        #facebook100 dataset 中所有以.mat结尾的文件
        graphs_list = [f[:-4] for f in os.listdir(path.join(data_dir, dataset_name)) if f.endswith(".mat") and not f.startswith("school")]
        print (len(graphs_list), graphs_list)
        feature_vals_all = np.empty((0, 6))
        for f in graphs_list:
            mat = scipy.io.loadmat(f'{data_dir}/{dataset_name}/{f}.mat')
            A = mat['A']
            metadata = mat['local_info']
            metadata = metadata.astype(int)
            feature_vals = np.hstack(
                (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
            feature_vals_all = np.vstack(
                (feature_vals_all, feature_vals)
            )
        
        random.shuffle(graphs_list)
        train_list = ['Amherst41', 'Caltech36', 'Johns Hopkins55']
        test_list = ['Bingham82', 'Duke14', 'Princeton12', 'WashU32', 'Brandeis99', 'Carnegie49', 'Cornell5', 'Yale4', 'Penn94', 'Brown11', 'Texas80']


        datalist = []
        for graph_name in train_list:
            print (graph_name)
            data = load_fb100(data_dir, graph_name, feature_vals_all)
            datalist.append(data)

        for graph_name in test_list:
            print (graph_name)
            data = load_fb100(data_dir, graph_name, feature_vals_all)
            datalist.append(data)
    
        graphs_list = train_list + test_list
        
        with open (path.join(data_dir, f"{dataset_name}/{dataset_name}.pkl"), 'wb') as f:
            pickle.dump([datalist, graphs_list], f) 

    elif dataset_name == "ogbn_arxiv":
        datalist = []
        graphs_list = []
        proportion = 1.0
        train_data = load_ogb_arxiv(data_dir, [1971, 2010], proportion)
        datalist.append(train_data)
        graphs_list.append([1971, 2009])

        interval = 1
        for i in range(2011, 2021, interval):
            data = load_ogb_arxiv(data_dir, [i, i+interval-1], proportion)
            datalist.append(data)
            graphs_list.append([i, i+interval-1])

        with open (path.join(data_dir, f"{dataset_name}/{dataset_name}.pkl"), 'wb') as f:
            pickle.dump([datalist, graphs_list], f) 

    elif dataset_name == "elliptic":
        datalist =[]
        graphs_list = []
        for i in range(0, 49):
            data = load_elliptic(data_dir, i)
            datalist.append(data)
            graphs_list.append(i)

        with open (path.join(data_dir, f"{dataset_name}/{dataset_name}.pkl"), 'wb') as f:
            pickle.dump([datalist, graphs_list], f) 

    return [datalist, graphs_list]



def load_twitch(data_dir, lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = f"{data_dir}/twitch/{lang}"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2]=="True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=int)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id:idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]
    
    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)), 
                                 (np.array(src), np.array(targ))),
                                shape=(n,n))
    features = np.zeros((n,3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    # features = features[:, np.sum(features, axis=0) != 0] # remove zero cols. not need for cross graph task
    new_label = label[reorder_node_ids]
    label = new_label

    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)

    data = Data(edge_index=edge_index, x=node_feat, y=label)
    
    return data


def load_fb100(data_dir, graph_name, feature_vals_all):
    mat = scipy.io.loadmat(f'{data_dir}/fb100/{graph_name}.mat')
    A = mat['A']
    metadata = mat['local_info']

    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled
    label = torch.tensor(label, dtype=torch.long)
    label = torch.where(label > 0, 1, 0)

    
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    '''
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        # feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        feat_onehot = label_binarize(feat_col, classes=np.unique(feature_vals_all[:, col]))
        features = np.hstack((features, feat_onehot))
    node_feat = torch.tensor(features, dtype=torch.float)
    '''
    node_feat = torch.tensor(feature_vals, dtype=torch.float)
    data = Data(edge_index=edge_index, x=node_feat, y=label)
    
    return data



def take_second(element):
    return element[1]


def load_ogb_arxiv(data_dir, year_bound = [2018, 2020], proportion = 1.0):
    import ogb.nodeproppred

    dataset = ogb.nodeproppred.NodePropPredDataset(name='ogbn-arxiv', root=data_dir)
    graph = dataset.graph

    node_years = graph['node_year']
    edges = graph['edge_index']
    n = node_years.shape[0]
    node_years = node_years.reshape(n)


    nodes_d = {}
    for i, year in enumerate(node_years):
        if year <= year_bound[1] and year >= year_bound[0]:
            nodes_d[i] = 0


    for i in range(edges.shape[1]):
        if edges[0][i] in nodes_d and edges[1][i] in nodes_d:
            nodes_d[edges[0][i]] += 1
            nodes_d[edges[1][i]] += 1
    
    nodes = [[k, v] for k, v in nodes_d.items()]
    nodes.sort(key = take_second, reverse = True)
    nodes = nodes[: int(proportion * len(nodes))]

    result_edges = []
    result_features = []

    for node in nodes:
        result_features.append(graph['node_feat'][node[0]])
    result_features = np.array(result_features)

    ids = {}
    for i, node in enumerate(nodes):
        ids[node[0]] = i

    for i in range(edges.shape[1]):
        if edges[0][i] in ids and edges[1][i] in ids:
            result_edges.append([ids[edges[0][i]], ids[edges[1][i]]])
    print (year_bound, len(nodes_d), len(result_edges))

    result_edges = np.array(result_edges).transpose(1, 0)

    result_labels = dataset.labels[[node[0] for node in nodes]]
    label = torch.tensor(result_labels, dtype=torch.long).squeeze(1)


    edge_index = torch.tensor(result_edges, dtype=torch.long)
    node_feat = torch.tensor(result_features, dtype=torch.float)

    data = Data(edge_index=edge_index, x=node_feat, y=label)
    
    return data

def load_elliptic(data_dir, i):
    with open (path.join(data_dir, f"elliptic/{i}.pkl"), 'rb') as f:
        A, label, features = pickle.load(f)
    
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)
    #print (label.shape, edge_index.shape)
    # number of each class in label
    #print (torch.bincount(label+1))

    mask = (label >= 0)

    data = Data(edge_index=edge_index, x=node_feat, y=label)
    data.mask = mask
    
    return data

