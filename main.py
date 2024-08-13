import torch
from torch_geometric.data import Data, Dataset, DataLoader, NeighborSampler, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
import argparse
import numpy as np
import scipy.sparse
import pickle
from model import *
from GCAL import *
from data_process import *
from validation import *
from tqdm import tqdm
import sys
import datetime
import random
import os


class CustomDataset(Dataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser.add_argument('--dataset', type=str, default='twitch', help='Dataset')
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset') #replace with your own data path
    parser.add_argument('--model', type=str, default='gcn', help='Model') 
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--device', type=str, default='0', help='Device')
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for deep methods')
    parser.add_argument('--show_interval', type=int, default=1, help='interval for showing training loss')
    parser.add_argument('--model_save_url', type=str, default='checkpoints', help='model save url')
    parser.add_argument('--resume', action='store_true', default=True, help='resume training')
    parser.add_argument('--gat_heads', type=int, default=8, help='number of heads for gat')
    parser.add_argument('--method', type=str, default='test', help='method: test, DA')
    parser.add_argument('--train_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--test_epochs', type=int, default=1, help='Number of epochs to test')
    parser.add_argument('--entropy_weight', type=float, default=0.1, help='Entropy weight for GCAL')
    parser.add_argument('--replay_weight', type=float, default=0.1, help='Replay weight for GCAL')
    parser.add_argument('--vae_weight', type=float, default=0.1, help='VAE weight for GCAL')
    parser.add_argument('--edge_weight', type=float, default=0.1, help='Edge weight for GCAL')
    parser.add_argument('--lr_model', type=float, default=0.01, help='Learning rate for the model in GCAL')
    parser.add_argument('--wd_model', type=float, default=0.0001, help='Weight decay for the model in GCAL')
    parser.add_argument('--lr_mem', type=float, default=0.01, help='Learning rate for memory in GCAL')
    parser.add_argument('--wd_mem', type=float, default=0.0001, help='Weight decay for memory in GCAL')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs in GCAL')
    parser.add_argument('--inner_loop', type=int, default=1, help='Number of inner loops in GCAL')
    parser.add_argument('--mt', type=float, default=0.5, help='MT parameter in GCAL')
    parser.add_argument('--syn_ratio', type=float, default=0, help='Synthetic data')
    args = parser.parse_args()
    print(args)
    
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    datalist, datanamelist = load_data(args)
    print ("datalist length: ", len(datalist))

    if args.dataset == "twitch":
        train_dataset = datalist[:1]
        test_dataset = datalist[1:]

    elif args.dataset == "fb100":
        train_dataset = datalist[:3]
        test_dataset = datalist[3:]
        datanamelist = ['0-2'] + datanamelist[3:]

    elif args.dataset == "ogbn-arxiv":
        train_dataset = datalist[:1]
        test_dataset = datalist[1:]

    elif args.dataset == "elliptic":
        train_dataset = datalist[6:9]
        test_dataset = datalist[9:]
        datanamelist = ['6-8'] + datanamelist[9:]
    print (datanamelist)
    
    num_labels = torch.max(train_dataset[0].y) + 1
    feature_dim = train_dataset[0].x.shape[1]
    print ("num_labels: ", num_labels)
    print ("feature_dim: ", feature_dim)
    args.feature_dim = feature_dim

    results_matrix = np.zeros((1+len(test_dataset), 1+len(test_dataset)))

    # ----------------- training ---------------------
    if args.resume or not os.path.exists(args.model_save_url+ f"/{args.dataset}_model_best.pt"): 
        if not os.path.exists(args.model_save_url):
            os.makedirs(args.model_save_url)
        merged_graph = Batch.from_data_list(train_dataset).to(args.device)
        print ("merged_train_graph: ", merged_graph) 
        if args.dataset == "elliptic":
            indices = torch.where(merged_graph.mask)[0] 
            indices = indices[torch.randperm(len(indices))] 
        else:    
            indices = torch.randperm(merged_graph.num_nodes)

        train_index = indices[:int(0.6 * len(indices))]
        val_index = indices[int(0.6 * len(indices)):int(0.8 * len(indices))]
        test_index = indices[int(0.8 * len(indices)):]

        train_mask, val_mask, test_mask = (torch.zeros(merged_graph.num_nodes, dtype=torch.bool) for _ in range(3))
        train_mask[train_index], val_mask[val_index], test_mask[test_index] = True, True, True
        merged_graph.train_mask, merged_graph.val_mask, merged_graph.test_mask = train_mask, val_mask, test_mask

        
        model = Model(args, feature_dim, num_labels).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_metric = -1
        for epoch in tqdm(range(args.train_epochs), ncols=100):
            model.train()
            
            optimizer.zero_grad()
            loss = model(merged_graph.x, merged_graph.edge_index, merged_graph.y, merged_graph.train_mask)
            loss.backward()
            optimizer.step()
            avg_loss = loss.item()
            if epoch % args.show_interval == 0:
                model.eval()
                pred = model.inference(merged_graph.x, merged_graph.edge_index)
                pred = pred.argmax(dim=1)
                metric = validation(args, pred, merged_graph.y, merged_graph, merged_graph.val_mask)
                if args.dataset == "twitch":
                    tqdm.write('Epoch: {:03d}, Loss: {:.4f}, auc: {:.4f}'.format(epoch, avg_loss, metric))
                elif args.dataset == "elliptic":
                    tqdm.write('Epoch: {:03d}, Loss: {:.4f}, f1: {:.4f}'.format(epoch, avg_loss, metric))
                else:
                    tqdm.write('Epoch: {:03d}, Loss: {:.4f}, acc: {:.4f}'.format(epoch, avg_loss, metric))
                if metric > best_metric:
                    best_epoch = epoch
                    best_metric = metric
                    test_metric = validation(args, pred, merged_graph.y, merged_graph, merged_graph.test_mask)
                    torch.save(model.state_dict(), args.model_save_url + f"/{args.dataset}_model_best.pt")

        print ("best epoch: {:03d}, best acc: {:.4f}".format(best_epoch, test_metric))
        


    # ----------------- testing ---------------------
    results_matrix[0,0] = '{:.6f}'.format(test_metric)
    model = Model(args, feature_dim, num_labels).to(args.device)
    try:
        model.load_state_dict(torch.load(args.model_save_url + f"/{args.dataset}_model_best.pt"))
    except FileNotFoundError:
        print(f"Model file not found in {args.model_save_url}")
        sys.exit(1)

    test_dataset = CustomDataset(test_dataset)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    cta_model = GCAL(args, model).to(args.device)
    source_graph = merged_graph
    for i, graph in enumerate(loader):
        target_graph = graph.to(args.device)
        output = cta_model.train(target_graph, epochs = args.test_epochs)
        cta_model.base_model.eval()
        with torch.no_grad():
            pre = cta_model.base_model.inference(source_graph.x, source_graph.edge_index)
            pre = pre.argmax(dim=1)
            metric_result = validation(args, pre, source_graph.y, source_graph, source_graph.test_mask)
            results_matrix[i+1,0] = '{:.6f}'.format(metric_result)
            for j in range(i+1):
                    graph = test_dataset[j].to(args.device)
                    pred = cta_model.base_model.inference(graph.x, graph.edge_index)
                    pred = pred.argmax(dim=1)
                    metric_result = validation(args, pred, graph.y, graph)
                    results_matrix[i+1,j+1] = '{:.6f}'.format(metric_result)

            mean_results = np.mean(results_matrix[i+1,:i+1+1])
            if args.dataset == "twitch":
                tqdm.write(str(datanamelist[i+1]).ljust(5) + ' mean_auc: {:.4f}'.format(mean_results))
            elif args.dataset == "elliptic":
                tqdm.write(str(datanamelist[i+1]).ljust(5) + ' mean_f1: {:.4f}'.format(mean_results))
            else:
                tqdm.write(str(datanamelist[i+1]).ljust(15) + ' mean_acc: {:.4f}'.format(mean_results))
    for i, results in enumerate(results_matrix):
        if i == 0:
            mean_gain = 0.0
        else: 
            mean_gain = -np.mean(np.diag(np.array(results_matrix))[:i] - results_matrix[i][:i]) 
        mean_results = np.mean(results_matrix[i,:i+1])
    print('mean_acc: {:.4f}'.format(mean_results) + '\n')
    print('mean_gain: {:.4f}'.format(mean_gain) + '\n')


    
    
    
    