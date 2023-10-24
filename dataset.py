import pandas as pd
import numpy as np
import random
import networkx as nx
from copy import deepcopy
import time
from sklearn import preprocessing
import sys


def LP_preprocessing(graphs, ratio_sample_pos_link):
    # graph with train and test edges
    graphs_complete = deepcopy(graphs)
    
    # collect negative test edges
    ls_test_edges_neg = []
    for idx, graph in enumerate(graphs):
        start = time.time()
        print('For graph {}, we need to collect {} negative edges.'.format(
            idx, int(graph.number_of_edges() * (ratio_sample_pos_link / 100))
        ))
        df_train_edges_pos = nx.to_pandas_edgelist(graph)
        df_test_edges_neg = pd.DataFrame(
            np.random.choice(list(graph.nodes()), 10 * graph.number_of_edges()), columns=['source']
        )
        df_test_edges_neg['target'] = np.random.choice(list(graph.nodes()), 10*graph.number_of_edges())
        df_test_edges_neg = df_test_edges_neg[df_test_edges_neg['source']<df_test_edges_neg['target']]
        df_test_edges_neg = df_test_edges_neg.drop_duplicates().reset_index(drop=True)
        df_test_edges_neg = pd.merge(
            df_train_edges_pos, df_test_edges_neg, indicator=True, how='outer'
        ).query('_merge=="right_only"').drop('_merge', axis=1).reset_index(drop=True)
        n_sample = int(graphs_complete[idx].number_of_edges() * (ratio_sample_pos_link / 100))
        test_edges_neg = random.sample(df_test_edges_neg.values.tolist(), n_sample)
        print('Generating {} negative instances uses {:.2f} seconds.'.format(idx, time.time()-start))
        ls_test_edges_neg.append(test_edges_neg)
    
    # collect positive edges
    ls_test_edges_pos = []
    for idx, graph in enumerate(graphs_complete):
        start = time.time()
        print('For graph {}, we need to remove {} edges.'.format(
            idx, int(graph.number_of_edges() * (ratio_sample_pos_link / 100))
        ))
        df_train_edges_pos = nx.to_pandas_edgelist(graph)
        G_train = nx.Graph(graphs[idx])
        edge_index = np.array(list(graph.edges))
        edges = np.transpose(edge_index)
        e = edges.shape[1]
        edges = edges[:, np.random.permutation(e)]
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))
        index_train = []
        index_val = []
        for i in range(e):
            node1 = edges[0,i]
            node2 = edges[1,i]
            if node_count[node1]>0 and node_count[node2] > 0:  # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e * ratio_sample_pos_link / 100):
                    break
            else:
                index_train.append(i)
        index_train = index_train + list(range(i + 1, e))
        edges_train = edges[:, index_train]
        edges_test = edges[:, index_val]
        test_edges_pos = [[edges_test[0, i], edges_test[1, i]] for i in range(edges_test.shape[1])]
        G_train.remove_edges_from(test_edges_pos)
        if len(test_edges_pos) < int(graph.number_of_edges() * (ratio_sample_pos_link / 100)):
            print('For graph {}, there are only {} positive instances.'.format(idx, len(test_edges_pos)))
            sys.exit("Can not remove more edges.")
        print('Generating {} positive instances uses {:.2f} seconds.'.format(idx, time.time()-start))
        graphs[idx] = G_train
        ls_test_edges_pos.append(test_edges_pos)
    
    # friends collections
    ls_df_friends = []
    for idx in range(len(graphs)):
        df_friends = nx.to_pandas_edgelist(graphs[idx])

        _x = deepcopy(df_friends)
        _x.columns = ['target', 'source']
        df_friends = pd.concat([df_friends, _x]).reset_index(drop=True)
        
        ls_df_friends.append(df_friends)
    
    # test and valid edges collections
    ls_valid_edges = []
    ls_test_edges = []
    for idx in range(len(graphs)):
        valid_edges_pos = random.sample(
            ls_test_edges_pos[idx], int(graphs_complete[idx].number_of_edges() * (ratio_sample_pos_link / 100) / 2)
        )
        valid_edges_neg = random.sample(
            ls_test_edges_neg[idx], int(graphs_complete[idx].number_of_edges() * (ratio_sample_pos_link / 100) / 2)
        )
        test_edges_pos = [item for item in ls_test_edges_pos[idx] if item not in valid_edges_pos]
        test_edges_neg = [item for item in ls_test_edges_neg[idx] if item not in valid_edges_neg]
        test_edges = {
            'positive': test_edges_pos,
            'negative': test_edges_neg
        }
        valid_edges = {
            'positive': valid_edges_pos,
            'negative': valid_edges_neg
        }
        ls_valid_edges.append(valid_edges)
        ls_test_edges.append(test_edges)

    return ls_df_friends, graphs_complete, graphs, ls_valid_edges, ls_test_edges


def get_dataset(dataset_name, use_features, task, ratio_sample: int = 0):
    if dataset_name == 'grid':
        print('is reading {} dataset...'.format(dataset_name))
        graph = nx.grid_2d_graph(20, 20)
        graph = nx.convert_node_labels_to_integers(graph)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        graph = nx.relabel_nodes(graph, mapping, copy=True)
        identify_oh_feature = np.identity(graph.number_of_nodes())
        graphs = [graph]
        features = [identify_oh_feature]
        print('datatset reading is done.')
    
    elif dataset_name == 'emails':
        print('is reading {} dataset...'.format(dataset_name))
        df = pd.read_csv('./data/emails/email.txt', header=None, sep=' ', names=['source', 'target'])
        graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

        df_label = pd.read_csv('./data/emails/email_labels.txt', header=None, sep=' ', names=['node_id', 'label'])
        df_label = df_label[df_label['label'].isin(df_label['label'].value_counts()[df_label['label'].value_counts()>20].index)]
        available_nodes = df_label['node_id'].unique()

        graph = graph.subgraph(available_nodes)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        graph = nx.relabel_nodes(graph, mapping, copy=True)

        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)
        # ecode label into numeric
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        identify_oh_feature = np.identity(graph.number_of_nodes())

        graphs = [graph]
        features = [identify_oh_feature]
        df_labels = [df_label]
    
    elif dataset_name == 'cora':
        print('is reading {} dataset...'.format(dataset_name))
        df = pd.read_csv('./data/cora/cora.cites', header=None, sep='\t', names=['source', 'target'])
        graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        graph = nx.relabel_nodes(graph, mapping, copy=True)

        # cora feature
        content = pd.read_csv('./data/cora/cora.content', header=None, sep='\t')
        df_feat = content[range(1434)].rename(columns={0: 'node_id'})
        df_label = content[[0, 1434]].rename(
            columns={
                0: 'node_id',
                1434: 'label'
            }
        )

        df_feat['node_id'] = df_feat['node_id'].replace(mapping)
        df_feat = df_feat.sort_values('node_id', ascending=True).reset_index(drop=True)

        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)
        # ecode label into numeric
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        graphs = [graph]
        if use_features:
            features = [df_feat[range(1, 1434)].values]
        else:
            identify_oh_feature = np.identity(graph.number_of_nodes())
            features = [identify_oh_feature]
        df_labels = [df_label]    
    
    elif dataset_name == 'citeseer':
        print('is reading {} dataset...'.format(dataset_name))
        df = pd.read_csv('./data/citeseer/citeseer.cites', header=None, sep='\t', names=['source', 'target'])
        graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

        # citeseer feature
        content = pd.read_csv('./data/citeseer/citeseer.content', header=None, sep='\t')
        content[0] = content[0].apply(str)
        available_nodes = content[0].unique()
        df_feat = content[range(3704)].rename(columns={0: 'node_id'})
        df_label = content[[0, 3704]].rename(
            columns={
                0: 'node_id',
                3704: 'label'
            }
        )

        graph = graph.subgraph(available_nodes)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        graph = nx.relabel_nodes(graph, mapping, copy=True)

        df_feat['node_id'] = df_feat['node_id'].replace(mapping)
        df_feat = df_feat.sort_values('node_id', ascending=True).reset_index(drop=True)

        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)
        # ecode label into numeric
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        graphs = [graph]
        if use_features:
            features = [df_feat[range(1, 1434)].values]
        else:
            identify_oh_feature = np.identity(graph.number_of_nodes())
            features = [identify_oh_feature]
        df_labels = [df_label]

    print(nx.info(graphs[0]))
    print('is processing dataset...')
    if task == 'LP':
        ls_df_friends, graphs_complete, graphs, ls_valid_edges, ls_test_edges = LP_preprocessing(
            graphs=graphs, ratio_sample_pos_link=ratio_sample
        )
        df_labels = 0
        print('data processing is done')
        return ls_df_friends, graphs_complete, graphs, ls_valid_edges, ls_test_edges, features, df_labels
    else:
        return graphs, features, df_labels
