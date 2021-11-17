import numpy as np
import random
import torch
import argparse
import sys

from dataset import get_dataset
from community_detection import hierarchical_structure_generation
from execution import execute_NC, execute_LP
from preparation import LP_preparation, NC_preparation

import warnings
warnings.filterwarnings('ignore')
###########################################################################

# sys.argv = ['']  # execution on jupyter notebook
parser = argparse.ArgumentParser()
# general
parser.add_argument('--comment', dest='comment', default='0', type=str,
                    help='comment')
parser.add_argument('--task', dest='task', default='LP', type=str,
                    help='LP; NC; Inductive')
parser.add_argument('--mode', dest='mode', default='baseline', type=str,
                    help='experiment mode. E.g., baseline or basemodel')
parser.add_argument('--model', dest='model', default='GCN', type=str,
                    help='model class name. E.g., GCN, PGNN, HCGNN...')
parser.add_argument('--dataset', dest='dataset', default='grid', type=str,
                    help='cora; grid; communities; ppi')
parser.add_argument('--gpu', dest='gpu', default=True, type=bool,
                    help='whether use gpu')
parser.add_argument('--SEED', dest='SEED', default=123, type=int)

# dataset
parser.add_argument('--ratio_sample_pos', dest='ratio_sample_pos', default=20, type=float)
parser.add_argument('--use_features', dest='use_features', default=True, type=bool,
                    help='whether use node features')
parser.add_argument('--community_detection_method', dest='community_detection_method', default='Louvain', type=str,
                    help='community detection method, default Louvain')
parser.add_argument('--threshold', dest='threshold', default=1, type=int,
                    help='the threshold for graph generation, default 1')

# model
parser.add_argument('--lr', dest='lr', default=1e-2, type=float)
parser.add_argument('--epoch_num', dest='epoch_num', default=201, type=int)
parser.add_argument('--epoch_log', dest='epoch_log', default=10, type=int)
parser.add_argument('--layer_num', dest='layer_num', default=2, type=int)
parser.add_argument('--relu', dest='relu', default=True, type=bool)
parser.add_argument('--dropout', dest='dropout', default=False, type=bool)
parser.add_argument('--drop_ratio', dest='drop_ratio', default=0.5, type=float)
parser.add_argument('--feature_pre', dest='feature_pre', default=True, type=bool)
parser.add_argument('--same_level_gnn', dest='same_level_gnn', default='GCN', type=str,
                    help='agg within level. E.g., MEAN GCN, SAGE, GAT, GIN, ...')
parser.add_argument('--down2up_gnn', dest='down2up_gnn', default='MEAN', type=str,
                    help='aggregation bottom-up. E.g., MEAN, GCN, SAGE, GAT, GIN, ...')
parser.add_argument('--up2down_gnn', dest='up2down_gnn', default='GAT', type=str,
                    help='aggregation top-down. E.g., GCN, SAGE, GAT, GIN, ...')
parser.add_argument('--fshot', dest='fshot', default=False, type=bool)

parser.set_defaults(gpu=False, task='LP', model='GCN', dataset='grid', feature_pre=True)
args = parser.parse_args()

SEED = args.SEED
np.random.seed(SEED)
random.seed(SEED)
device = torch.device('cuda:'+str(0) if args.gpu and torch.cuda.is_available() else 'cpu')


if args.task == 'LP':
    ls_df_friends, graphs_complete, graphs, ls_valid_edges, ls_test_edges, features, df_labels = get_dataset(
        dataset_name=args.dataset,
        use_features=args.use_features,
        task=args.task,
        ratio_sample_pos=args.ratio_sample_pos
    )
    ls_hierarchical_community, ls_up2down_edges, ls_down2up_edges = hierarchical_structure_generation(
        dataset_name=args.dataset,
        graphs=graphs,
        method=args.community_detection_method,
        threshold=args.threshold
    )
    ls_adj_same_level, ls_df_train, ls_df_valid, ls_df_test = LP_preparation(
        graphs=graphs,
        ls_df_friends=ls_df_friends,
        ls_test_edges=ls_test_edges,
        ls_valid_edges=ls_valid_edges,
        ls_hierarchical_community=ls_hierarchical_community
    )
    execute_LP(
        args, graphs, features, ls_hierarchical_community,
        ls_adj_same_level, ls_up2down_edges, ls_down2up_edges,
        ls_df_train, ls_df_valid, ls_df_test, device
    )
else:
    graphs, ls_train_nodes, ls_valid_nodes, ls_test_nodes, features, df_labels = get_dataset(
        dataset_name=args.dataset,
        use_features=args.use_features,
        task=args.task,
        ratio_sample_pos=args.ratio_sample_pos
    )
    ls_hierarchical_community, ls_up2down_edges, ls_down2up_edges = hierarchical_structure_generation(
        dataset_name=args.dataset,
        graphs=graphs,
        method=args.community_detection_method,
        threshold=args.threshold
    )
    ls_adj_same_level = NC_preparation(
        graphs=graphs,
        ls_hierarchical_community=ls_hierarchical_community
    )
    execute_NC(
        args, graphs, df_labels, features, ls_hierarchical_community,
        ls_adj_same_level, ls_up2down_edges, ls_down2up_edges,
        ls_train_nodes, ls_valid_nodes, ls_test_nodes, device
    )
