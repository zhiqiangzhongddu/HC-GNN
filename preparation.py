import numpy as np
import networkx as nx
import pandas as pd
import random
from copy import deepcopy
import torch
import torch_geometric as tg


def graph_to_adj(graphs, ls_hierarchical_community):
    # add generated graphs into edges
    ls_adj_same_level = []
    for idx, graph in enumerate(graphs):
        G_same_level = deepcopy(graph)
        hierarchical_community = ls_hierarchical_community[idx]
        
        add_nodes = []
        add_edges = []
        for com in hierarchical_community:
            add_nodes += com['partitions']
            add_edges += com['edges']
        G_same_level.add_nodes_from(add_nodes)
        G_same_level.add_edges_from(add_edges)
        adj_same_level = nx.to_scipy_sparse_matrix(G_same_level)
        
        ls_adj_same_level.append(adj_same_level)
    return ls_adj_same_level


def set_up_train_test_valid(graphs, ls_df_friends, ls_valid_edges, ls_test_edges, seed=123):
    # valid data
    ls_df_valid = []
    for idx, valid_edges in enumerate(ls_valid_edges):
        df_valid_pos_samples = pd.DataFrame(valid_edges['positive'], columns=['source', 'target'])
        df_valid_pos_samples['label'] = 1
        df_valid_neg_samples = pd.DataFrame(valid_edges['negative'], columns=['source', 'target'])
        df_valid_neg_samples['label'] = 0

        df_valid = pd.concat([df_valid_pos_samples, df_valid_neg_samples], axis=0)
        
        ls_df_valid.append(df_valid)
    # test data
    ls_df_test = []
    for idx, test_edges in enumerate(ls_test_edges):
        df_test_pos_samples = pd.DataFrame(test_edges['positive'], columns=['source', 'target'])
        df_test_pos_samples['label'] = 1
        df_test_neg_samples = pd.DataFrame(test_edges['negative'], columns=['source', 'target'])
        df_test_neg_samples['label'] = 0

        df_test = pd.concat([df_test_pos_samples, df_test_neg_samples], axis=0)
        
        ls_df_test.append(df_test)
    # train data
    ls_df_train = []
    for idx, friends in enumerate(ls_df_friends):
        graph = graphs[idx]
        df_train_neg = pd.DataFrame(
            np.random.choice(list(graph.nodes()), 10 * graph.number_of_edges()), columns=['source']
        )
        df_train_neg['target'] = np.random.choice(list(graph.nodes()), 10*graph.number_of_edges())
        df_train_neg = df_train_neg[df_train_neg['source']<df_train_neg['target']]
        df_train_neg = df_train_neg.drop_duplicates().reset_index(drop=True)
        
        df_valid = ls_df_valid[idx]
        df_test = ls_df_test[idx]
        df_train_pos = ls_df_friends[idx]
        df_train_pos = friends[friends['source']<friends['target']]
        df_train_pos['label'] = 1
        df_non = pd.concat([df_train_pos, df_valid, df_test]).reset_index(drop=True)[['source', 'target']]
        
        df_train_neg = pd.merge(
            df_non, df_train_neg, indicator=True, how='outer'
        ).query('_merge=="right_only"').drop('_merge', axis=1).reset_index(drop=True)
        df_train_neg = df_train_neg.sample(df_train_pos.shape[0], random_state=seed)
        df_train_neg['label'] = 0
        
        df_train = pd.concat([df_train_pos, df_train_neg]).drop_duplicates().reset_index(drop=True)
        ls_df_train.append(df_train)
    return ls_df_train, ls_df_valid, ls_df_test


def LP_preparation(graphs, ls_df_friends, ls_test_edges, ls_valid_edges, ls_hierarchical_community):
    print('is preparing datasets...')
    ls_adj_same_level = graph_to_adj(graphs=graphs, ls_hierarchical_community=ls_hierarchical_community)
    ls_df_train, ls_df_valid, ls_df_test = set_up_train_test_valid(
        graphs=graphs, ls_df_friends=ls_df_friends, ls_valid_edges=ls_valid_edges, ls_test_edges=ls_test_edges
    )

    print('dataset preparation is done')
    return ls_adj_same_level, ls_df_train, ls_df_valid, ls_df_test


def NC_preparation(graphs, ls_hierarchical_community):
    print('is preparing datasets...')
    ls_adj_same_level = graph_to_adj(graphs=graphs, ls_hierarchical_community=ls_hierarchical_community)
    
    print('dataset preparation is done')
    return ls_adj_same_level


def LP_set_up(
        config, graphs, features,
        ls_hierarchical_community, ls_adj_same_level, ls_up2down_edges, ls_down2up_edges,
        ls_df_train, ls_df_valid, ls_df_test, device
):
    # set up train, valid and test data
    ls_train_user_left = []
    ls_train_user_right = []
    ls_train_labels = []
    ls_train_labels_tensor = []
    ls_valid_user_left = []
    ls_valid_user_right = []
    ls_valid_labels = []
    ls_test_user_left = []
    ls_test_user_right = []
    ls_test_labels = []

    for idx in range(len(graphs)):
        ls_train_user_left.append(ls_df_train[idx]['source'].values.tolist())
        ls_train_user_right.append(ls_df_train[idx]['target'].values.tolist())
        ls_train_labels.append(ls_df_train[idx]['label'].values.tolist())
        ls_train_labels_tensor.append(
            torch.tensor(ls_df_train[idx]['label'].values.tolist(), dtype=torch.float).to(device)
        )
        
        ls_valid_user_left.append(ls_df_valid[idx]['source'].values.tolist())
        ls_valid_user_right.append(ls_df_valid[idx]['target'].values.tolist())
        ls_valid_labels.append(ls_df_valid[idx]['label'].values.tolist())

        ls_test_user_left.append(ls_df_test[idx]['source'].values.tolist())
        ls_test_user_right.append(ls_df_test[idx]['target'].values.tolist())
        ls_test_labels.append(ls_df_test[idx]['label'].values.tolist())
    # prepare features
    for idx, feature in enumerate(features):
        to_add = np.zeros((ls_adj_same_level[idx].shape[0]-feature.shape[0], feature.shape[1]), dtype=feature.dtype)
        feature = np.append(feature, to_add, axis=0)
        
        features[idx] = torch.FloatTensor(feature).to(device)
    # set up data
    ls_data = []

    for idx, graph in enumerate(graphs):
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1,0)
        
        x = features[idx]
        data = tg.data.Data(x=x, edge_index=edge_index).to(device)
        ls_data.append(data)
    # edges in same level
    ls_same_level_edges_index = []
    for idx, adj_same_level in enumerate(ls_adj_same_level):
        same_level_edges_tuples = np.where(adj_same_level.toarray() == 1)
        same_level_edges_index = [(same_level_edges_tuples[0][idx], same_level_edges_tuples[1][idx])
                                  for idx in range(len(same_level_edges_tuples[0]))]
        ls_same_level_edges_index.append(same_level_edges_index)
        
    # edges from Up to Down
    ls_up2down_edges_index = []
    for idx, up2down_edges in enumerate(ls_up2down_edges):
        up2down_edges_index = []
        for key, values in up2down_edges.items():
            for value in values:
                up2down_edges_index.append((key, value))
        ls_up2down_edges_index.append(up2down_edges_index)
    
    # edges from Down to Up
    if config.down2up_gnn == 'MEAN':
        ls_down2up_arrays = []
        ls_down2up_edges_index = None
        for idx, adj_same_level in enumerate(ls_adj_same_level):
            down2up_edges = ls_down2up_edges[idx]
            down2up_array = np.zeros([adj_same_level.shape[0], adj_same_level.shape[0]])

            for key, values in down2up_edges.items():
                for value in values:
                    down2up_array[value, key] = 1

            ls_tmp = []
            hierarchical_community = ls_hierarchical_community[idx]
            for idc, community in enumerate(hierarchical_community):
                if idc == 0:
                    tmp = deepcopy(down2up_array)
                    tmp[:min(community['partitions']), min(community['partitions']):] = 0
                    tmp[max(community['partitions'])+1:, :] = 0
                else:
                    tmp = deepcopy(down2up_array)
                    tmp[:min(community['partitions']), :min(hierarchical_community[idc-1]['partitions'])] = 0
                    tmp[max(community['partitions'])+1:, max(hierarchical_community[idc-1]['partitions']):] = 0
                ls_tmp.append(tmp)

            ls_down2up_arrays.append(ls_tmp)
    else:
        ls_down2up_arrays = None
        ls_down2up_edges_index = []
        for idx, down2up_edges in enumerate(ls_down2up_edges):
            down2up_edges_index = []
            for key, values in down2up_edges.items():
                for value in values:
                    down2up_edges_index.append((key, value))
            ls_down2up_edges_index.append(down2up_edges_index)
    

    ls_data = []
    ls_data_up2down = []
    ls_data_down2up = []
    ls_down2up_torch_arrays = []

    for idx in range(len(graphs)):
        same_level_edges_index = ls_same_level_edges_index[idx]
        feature = features[idx]
        up2down_edges_index = ls_up2down_edges_index[idx]
        if config.down2up_gnn == 'MEAN':
            down2up_arrays = ls_down2up_arrays[idx]
            down2up_edges_index = None
        else:
            down2up_arrays = None
            down2up_edges_index = ls_down2up_edges_index[idx]
        
        edge_index = torch.tensor(same_level_edges_index, dtype=torch.long)
        data = tg.data.Data(x=feature, edge_index=edge_index.t().contiguous()).to(device)
        ls_data.append(data)

        up2down_edges_index = torch.tensor(up2down_edges_index, dtype=torch.long)
        data_up2down = tg.data.Data(x=feature, edge_index=up2down_edges_index.t().contiguous()).to(device)
        ls_data_up2down.append(data_up2down)
        
        if config.down2up_gnn == 'MEAN':
            down2up_torch_arrays = [torch.tensor(down2up_array, dtype=torch.float).to(device)
                                    for down2up_array in down2up_arrays]
            ls_down2up_torch_arrays.append(down2up_torch_arrays)
        else:
            down2up_edges_index = torch.tensor(down2up_edges_index, dtype=torch.long)
            data_down2up = tg.data.Data(
                x=feature, edge_index=down2up_edges_index.t().contiguous()
            ).to(device)
            ls_data_down2up.append(data_down2up)
    
    # experiments
    ls_train_user_left = [torch.LongTensor(train_user_left).to(device) for train_user_left in ls_train_user_left]
    ls_train_user_right = [torch.LongTensor(train_user_right).to(device) for train_user_right in ls_train_user_right]
    ls_valid_user_left = [torch.LongTensor(valid_user_left).to(device) for valid_user_left in ls_valid_user_left]
    ls_valid_user_right = [torch.LongTensor(valid_user_right).to(device) for valid_user_right in ls_valid_user_right]
    ls_test_user_left = [torch.LongTensor(test_user_left).to(device) for test_user_left in ls_test_user_left]
    ls_test_user_right = [torch.LongTensor(test_user_right).to(device) for test_user_right in ls_test_user_right]

    # for item in [ls_data, ls_data_up2down, ls_data_down2up, ls_down2up_torch_arrays, ls_train_user_left, ls_train_user_right, ls_valid_user_left, ls_valid_user_right, ls_test_user_left, ls_test_user_right, ls_train_labels, ls_train_labels_tensor, ls_valid_labels, ls_test_labels]:
    #     print(len(item))
    print('data set up is done')
    return ls_data, ls_data_up2down, ls_data_down2up, ls_down2up_torch_arrays, ls_train_user_left, ls_train_user_right, ls_valid_user_left, ls_valid_user_right, ls_test_user_left, ls_test_user_right, ls_train_labels, ls_train_labels_tensor, ls_valid_labels, ls_test_labels


def NC_set_up(
        config, graphs, df_labels, features,
        ls_hierarchical_community, ls_adj_same_level, ls_up2down_edges, ls_down2up_edges, device
):
    for idx, feature in enumerate(features):
        features[idx] = np.array(list(feature))
    # set up semi-sup, few-shot
    ls_train_nodes = []
    ls_valid_nodes = []
    ls_test_nodes = []
    if config.dataset in ['emails', 'communities']:
        for idx in range(len(graphs)):
            if config.fshot:
                ls_train_nodes.append(
                    np.hstack(
                        df_labels[idx].groupby('label')['node_id'].apply(list).apply(
                            lambda items: random.sample(items, 5)).values
                    )
                )
            else:
                ls_train_nodes.append(
                    np.hstack(
                        df_labels[idx].groupby('label')['node_id'].apply(list).apply(
                            lambda items: random.sample(items, 20)).values
                    )
                )
            non_train_nodes = random.sample(
                np.array([node for node in graphs[idx].nodes() if node not in ls_train_nodes[-1]]),
                int(graphs[idx].number_of_nodes() * 0.20)
            )
            ls_valid_nodes.append(
                np.array(non_train_nodes[: int(graphs[idx].number_of_nodes() * 0.1)])
            )
            ls_test_nodes.append(
                np.array(non_train_nodes[int(graphs[idx].number_of_nodes() * 0.1):])
            )
    else:
        for idx in range(len(graphs)):
            if config.fshot:
                ls_train_nodes.append(
                    np.hstack(
                        df_labels[idx].groupby('label')['node_id'].apply(list).apply(
                            lambda items: random.sample(items, 5)).values
                    )
                )
            else:
                ls_train_nodes.append(
                    np.hstack(
                        df_labels[idx].groupby('label')['node_id'].apply(list).apply(
                            lambda items: random.sample(items, 20)).values
                    )
                )

            ls_valid_nodes.append(
                np.array(
                    random.sample(set(graphs[idx].nodes()) - set(ls_train_nodes[idx]), 500)
                )
            )
            ls_test_nodes.append(
                np.array(
                    random.sample(set(graphs[idx].nodes()) - set(ls_train_nodes[idx]) - set(ls_valid_nodes[idx]), 1000)
                )
            )
    for idx in range(len(graphs)):
        print('for graph-{}, there are {} train, {} valid and {} test nodes.'.format(
            idx, ls_train_nodes[idx].shape[0], ls_valid_nodes[idx].shape[0], ls_test_nodes[idx].shape[0]
        ))

    # prepare features
    for idx, feature in enumerate(features):
        to_add = np.zeros((ls_adj_same_level[idx].shape[0]-feature.shape[0], feature.shape[1]), dtype=feature.dtype)
        feature = np.append(feature, to_add, axis=0)
        
        features[idx] = torch.FloatTensor(feature).to(device)
    # set up data
    ls_data = []

    for idx, graph in enumerate(graphs):
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:,::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1,0)
        
        x = features[idx]
        data = tg.data.Data(x=x, edge_index=edge_index).to(device)
        ls_data.append(data)
    
    # prepare labels
    ls_labels_tensor = []
    ls_labels = []
    for idx, df_label in enumerate(df_labels):
        
        ls_labels_tensor.append(torch.LongTensor(df_label['label'].values).to(device))
        ls_labels.append(pd.get_dummies(df_label['label']).values)
    
    # edges in same level
    ls_same_level_edges_index = []
    for idx, adj_same_level in enumerate(ls_adj_same_level):
        same_level_edges_tuples = np.where(adj_same_level.toarray()==1)
        same_level_edges_index = [(same_level_edges_tuples[0][idx], same_level_edges_tuples[1][idx])
                                  for idx in range(len(same_level_edges_tuples[0]))]
        ls_same_level_edges_index.append(same_level_edges_index)
    del idx, adj_same_level, same_level_edges_index
        
    # edges from Up to Down
    ls_up2down_edges_index = []
    for idx, up2down_edges in enumerate(ls_up2down_edges):
        up2down_edges_index = []
        for key, values in up2down_edges.items():
            for value in values:
                up2down_edges_index.append((key, value))
        ls_up2down_edges_index.append(up2down_edges_index)
    del idx, up2down_edges

    # edges from Down to Up
    if config.down2up_gnn == 'MEAN':
        ls_down2up_arrays = []
        for idx, adj_same_level in enumerate(ls_adj_same_level):
            down2up_edges = ls_down2up_edges[idx]
            down2up_array = np.zeros([adj_same_level.shape[0], adj_same_level.shape[0]])

            for key, values in down2up_edges.items():
                for value in values:
                    down2up_array[value, key] = 1

            ls_tmp = []
            hierarchical_community = ls_hierarchical_community[idx]
            for idc, community in enumerate(hierarchical_community):
                if idc == 0:
                    tmp = deepcopy(down2up_array)
                    tmp[:min(community['partitions']), min(community['partitions']):] = 0
                    tmp[max(community['partitions'])+1:, :] = 0
                else:
                    tmp = deepcopy(down2up_array)
                    tmp[:min(community['partitions']), :min(hierarchical_community[idc-1]['partitions'])] = 0
                    tmp[max(community['partitions'])+1:, max(hierarchical_community[idc-1]['partitions']):] = 0
                ls_tmp.append(tmp)

            ls_down2up_arrays.append(ls_tmp)
    else:
        ls_down2up_edges_index = []
        for idx, down2up_edges in enumerate(ls_down2up_edges):
            down2up_edges_index = []
            for key, values in down2up_edges.items():
                for value in values:
                    down2up_edges_index.append((key, value))
            ls_down2up_edges_index.append(down2up_edges_index)
        del idx, down2up_edges

    ls_data = []
    ls_data_up2down = []
    ls_data_down2up = []
    ls_down2up_torch_arrays = []

    for idx in range(len(graphs)):
        same_level_edges_index = ls_same_level_edges_index[idx]
        feature = features[idx]
        up2down_edges_index = ls_up2down_edges_index[idx]
        if config.down2up_gnn == 'MEAN':
            down2up_arrays = ls_down2up_arrays[idx]
        else:
            down2up_edges_index = ls_down2up_edges_index[idx]
        
        edge_index = torch.tensor(same_level_edges_index, dtype=torch.long)
        data = tg.data.Data(
            x=feature, edge_index=edge_index.t().contiguous()
        ).to(device)
        ls_data.append(data)

        up2down_edges_index = torch.tensor(up2down_edges_index, dtype=torch.long)
        data_up2down = tg.data.Data(
            x=feature, edge_index=up2down_edges_index.t().contiguous()
        ).to(device)
        ls_data_up2down.append(data_up2down)
        
        if config.down2up_gnn == 'MEAN':
            down2up_torch_arrays = [torch.tensor(down2up_array, dtype=torch.float).to(device)
                                    for down2up_array in down2up_arrays]
            ls_down2up_torch_arrays.append(down2up_torch_arrays)
        else:
            down2up_edges_index = torch.tensor(down2up_edges_index, dtype=torch.long)
            data_down2up = tg.data.Data(x=feature, edge_index=down2up_edges_index.t().contiguous()).to(device)
            ls_data_down2up.append(data_down2up)

    print('data set up is done')
    return ls_data, features, ls_data_up2down, ls_data_down2up, ls_down2up_torch_arrays, ls_train_nodes, ls_valid_nodes, ls_test_nodes, ls_labels, ls_labels_tensor
