"""Data utils functions for pre-processing and data loading."""
import os
import torch
import numpy as np
import networkx as nx
import torch_geometric.transforms as T
import random
import os.path as osp

import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

def order_edge_index(edge_index):
    row1, row2 = edge_index[0], edge_index[1]
    mask = row1 > row2
    edge_index[0, mask], edge_index[1, mask] = edge_index[1, mask], edge_index[0, mask]
    
    return edge_index

def load_data(dataset_str):
    data = np.load(os.path.join('heterophilic-data', f'{dataset_str.replace("-", "_")}.npz'))
    features = torch.tensor(data['node_features'])
    labels = torch.tensor(data['node_labels'])
    edges = torch.tensor(data['edges']).t()
    full_edges = torch.unique(torch.cat([edges, edges.flip(0)], dim=1), dim=1)
    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])

    dataset = Data(x=features,
                   edge_index=full_edges if dataset_str != 'roman_empire' else edges,
                   y=labels,
                   train_mask=train_masks,
                   val_mask=val_masks,
                   test_mask=test_masks)

    return dataset

def load_synthetic_data(task):
    N = 24
    size = 3
    datasets = []
    if task == 'oversquashing':
        graph = nx.barbell_graph(6, 0)
    elif task == 'oversmoothing':
        graph = nx.complete_graph(12)
    else:
        raise ValueError(f"Unknown task: {task}")
    edges = torch.tensor(list(graph.edges)).t()
    full_edges = torch.unique(torch.cat([edges, edges.flip(0)], dim=1), dim=1)
    for _ in range(100):
        features = torch.zeros((graph.number_of_nodes(), size))
        for i in range(graph.number_of_nodes()//2):
            features[i] = torch.tensor(np.random.uniform(-(3*N)**0.5, 0, size))
            features[i+graph.number_of_nodes()//2] = torch.tensor(np.random.uniform(0, (3*N)**0.5, size))
        
        avg_left = features[:graph.number_of_nodes()//2].mean(dim=0) #/ N
        avg_right = features[graph.number_of_nodes()//2:].mean(dim=0) #/ N
        
        labels = torch.zeros((graph.number_of_nodes(), size))
        labels[:graph.number_of_nodes()//2] = avg_right
        labels[graph.number_of_nodes()//2:] = avg_left

        datasets.append(Data(
            x=features,
            edge_index=full_edges,
            y=labels,
            random_walk_pe=torch.empty(graph.number_of_nodes(), 0)
        ))
    
    return datasets

def load_homophilic_data(dataset_str):
    data_root = osp.join(ROOT_DIR, 'datasets')
    data = Planetoid(root=data_root, name=dataset_str, split="geom-gcn", transform=T.NormalizeFeatures())[0]
    data.train_mask = data.train_mask.t().to(torch.bool)
    data.val_mask = data.val_mask.t().to(torch.bool)
    data.test_mask = data.test_mask.t().to(torch.bool)
    return data

# def load_synthetic_data(p=5, num_graphs=5000):
#     graph = nx.cycle_graph(10)
#     edges = torch.tensor(list(graph.edges)).t()
#     full_edges = torch.unique(torch.cat([edges, edges.flip(0)], dim=1), dim=1)
#     source = 0
#     target = 5
#     datasets = []
#     for _ in range(num_graphs):
#         one_hot_index = np.random.randint(p)
#         one_hot_vector = torch.nn.functional.one_hot(torch.tensor(one_hot_index), num_classes=p).float()
#         features = torch.zeros((10, p))
#         features[source] = torch.zeros(p)
#         features[target] = one_hot_vector
#         for i in range(10):
#             if i != source and i != target:
#                 features[i] = torch.ones(p)
#         labels = torch.zeros((10, p))
#         labels[source] = features[target]
#         datasets.append(Data(
#             x=features,
#             edge_index=full_edges,
#             y=labels,
#             random_walk_pe=torch.empty(graph.number_of_nodes(), 0)
#         ))
#     return datasets