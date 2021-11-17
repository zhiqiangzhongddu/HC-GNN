import pandas as pd
from collections import defaultdict
from copy import deepcopy
import community
from networkx.algorithms.community.centrality import girvan_newman
import itertools

from utils import max_lists, min_lists, generate_bigram, parallel_set_edge_between_community


def Louvain_community_detection(graphs):
    print('Is doing community detection....')
    # community_detection
    ls_hierarchical_community = []

    for idx, G in enumerate(graphs):
        print('start {} subgraph...'.format(idx))
        dendrogram_old = community.generate_dendrogram(G)
        dendrogram_new = deepcopy(dendrogram_old)
        hierarchical_community = []

        for level in range(len(dendrogram_old)):
            community_tmp = {}
            print('iteration {}: from {} items to {} items, modularity is {}.'.format(
                level, len(dendrogram_old[level]), len(set(dendrogram_old[level].values())),
                community.modularity(
                    community.partition_at_level(dendrogram_old, level), graph=G
                )
            ))
            community_tmp['before_num_partitions'] = len(dendrogram_old[level])
            community_tmp['after_num_partitions'] = len(set(dendrogram_old[level].values()))
            community_tmp['modularity'] = community.modularity(
                community.partition_at_level(dendrogram_old, level), graph=G
            )

            if level == 0:
                new_update = defaultdict(list)
                for key, value in dendrogram_new[level].items():
                    new_update[value+max(dendrogram_new[level])+1].append(key)
                dendrogram_new[level] = new_update
                community_tmp['edges_to_lower'] = new_update
                community_tmp['edges_to_lowest'] = new_update
            else:
                # edges to lower
                new_update = defaultdict(list)
                for key, value in dendrogram_new[level].items():
                    new_update[value+max(dendrogram_new[level-1])+1].append(
                        key + max_lists(dendrogram_new[level - 1].values()) + 1
                    )
                dendrogram_new[level] = new_update
                community_tmp['edges_to_lower'] = new_update       
                # edges to lowest
                new_update_lowest = deepcopy(new_update)
                for key, values in new_update.items():
                    new_values = []
                    for value in values:
                        new_values += hierarchical_community[level - 1]['edges_to_lowest'][value]
                    new_update_lowest[key] = new_values
                community_tmp['edges_to_lowest'] = new_update_lowest
            community_tmp['partitions'] = list(dendrogram_new[level].keys())

            hierarchical_community.append(community_tmp)
        ls_hierarchical_community.append(hierarchical_community)
    print('Community detection is done')
    return ls_hierarchical_community


def GN_community_detection(graphs, dataset_name):
    k = {'emails': 2,
        'grid': 4,
        'cora': 4,
        'power': 5,
        'citeseer': 4,
        'pubmed': 4}[dataset_name]

    ls_hierarchical_community = []

    for idx, G in enumerate(graphs):
        print('start {} subgraph...'.format(idx))
        G = graphs[0]
        comp = girvan_newman(G)
        ls_community = []
        for idx, communities in enumerate(itertools.islice(comp, k)):
            ls_community.append(communities)
        ls_community = ls_community[::-1] # we need a pyramidal structure

        hierarchical_community = []
        for level, community in enumerate(ls_community):
            community_tmp = {}
            if level == 0:
                community_tmp['before_num_partitions'] = G.number_of_nodes()
                community_tmp['after_num_partitions'] = len(community)
                new_update = defaultdict(list)
                for idx, com in enumerate(community):
                    new_update[idx+max(G.node)+1] = list(com)
                community_tmp['edges_to_lower'] = new_update
                community_tmp['edges_to_lowest'] = new_update
            else:
                community_tmp['before_num_partitions'] = hierarchical_community[-1]['after_num_partitions']
                community_tmp['after_num_partitions'] = len(community)
                # edges to lowest
                new_update_lowest = defaultdict(list)
                for idx, com in enumerate(community):
                    new_update_lowest[idx+max(hierarchical_community[-1]['partitions'])+1] = list(com)
                # edges to lower
                new_update_lower = defaultdict(list)
                for idx, com in enumerate(community):
                    for key, values in hierarchical_community[-1]['edges_to_lowest'].items():
                        if set(values).issubset(com):
                            new_update_lower[idx+max(hierarchical_community[-1]['partitions'])+1].append(key)
                community_tmp['edges_to_lower'] = new_update_lower
                community_tmp['edges_to_lowest'] = new_update_lowest

            community_tmp['partitions'] = list(community_tmp['edges_to_lowest'].keys())    
            hierarchical_community.append(community_tmp)
        ls_hierarchical_community.append(hierarchical_community)
    return ls_hierarchical_community


def present_community_detection_results(ls_hierarchical_community):
    print('is presenting hierarchical structure....')
    for idx, hierarchical_community in enumerate(ls_hierarchical_community):
        print('start graph community {}'.format(idx))
        for i in range(len(hierarchical_community)):
            print('layer {}: keys: [{}, {}], values: [{}, {}]'.format(
                i, min(hierarchical_community[i]['edges_to_lower']), 
                max(hierarchical_community[i]['edges_to_lower']),
                min_lists(hierarchical_community[i]['edges_to_lower'].values()),
                max_lists(hierarchical_community[i]['edges_to_lower'].values())))


def up2down_pipeline(graphs, ls_hierarchical_community, threshold):
    print('Is setting up hierarchical pipelines....')
    # set up up-down pipelline
    for idx, hierarchical_community in enumerate(ls_hierarchical_community):
        graph = graphs[idx]
        print('start graph {}'.format(idx))
        for id_community, community in enumerate(hierarchical_community):
            df_all_bigrams = pd.DataFrame({'all_bigrams': generate_bigram(community['partitions'])})
            df_all_bigrams = parallel_set_edge_between_community(
                graph=graph, community=community, df=df_all_bigrams, threshold=threshold
            )
            edges_tmp = df_all_bigrams[df_all_bigrams['result']]['all_bigrams'].values.tolist()
            hierarchical_community[id_community]['edges'] = edges_tmp
        ls_hierarchical_community[idx] = hierarchical_community
    # # verify the correctness of up-down pipelline
    # for idx, hierarchical_community in enumerate(ls_hierarchical_community):
    #     if len(hierarchical_community)==4:
    #         for key, values in hierarchical_community[3]['edges_to_lower'].items():
    #             res = []
    #             third_keys = []
    #             for sec_key in values:
    #                 third_keys += hierarchical_community[2]['edges_to_lower'][sec_key]
    #             fourth_keys = []
    #             for third_key in third_keys:
    #                 fourth_keys += hierarchical_community[1]['edges_to_lower'][third_key]
    #             for fourth_key in fourth_keys:
    #                 res += hierarchical_community[0]['edges_to_lower'][fourth_key]
    #             if res != hierarchical_community[3]['edges_to_lowest'][key]:
    #                 print(key, res, hierarchical_community[3]['edges_to_lowest'][key])
    print('Hierarchical pipelines are ready')
    return ls_hierarchical_community


def up2down_edges(ls_hierarchical_community):
    print('is recording up2down edges....')
    # all above layers messages pass to first layer
    ls_up2down_edges = []
    ls_up2down_dicts = []
    for idx, hierarchical_community in enumerate(ls_hierarchical_community):
        up2down_edges = defaultdict(list)
        up2down_dicts = defaultdict(list)

        for com in hierarchical_community:
            up2down_edges.update(com['edges_to_lowest'])
        ls_up2down_edges.append(dict(up2down_edges))
        
        for com in hierarchical_community:
            for (key, values) in com['edges_to_lowest'].items():
                for value in values:
                    up2down_dicts[value].append(key)
        # set high level order
        for (key, values) in up2down_dicts.items():
            up2down_dicts[key] = sorted(values)
        ls_up2down_dicts.append(dict(up2down_dicts))

    for idx, up2down_edges in enumerate(ls_up2down_edges):
        print('keys in [{}, {}], values in [{}, {}]'.format(
            min(up2down_edges.keys()),
            max(up2down_edges.keys()),
            min([value for values in up2down_edges.values() for value in values]),
            max([value for values in up2down_edges.values() for value in values])
        ))
    return ls_up2down_edges, ls_up2down_dicts


def down2up_edges(ls_hierarchical_community):
    print('is recording down2up edges....')
    # Down to the above one
    ls_down2up_edges = []
    for idx, hierarchical_community in enumerate(ls_hierarchical_community):
        down2up_edges = defaultdict(list)

        for com in hierarchical_community:
            for key, values in com['edges_to_lower'].items():
                for value in values:
                    down2up_edges[value].append(key)
        
        ls_down2up_edges.append(dict(down2up_edges))
    # verify down2up
    for idx, down2up_edges in enumerate(ls_down2up_edges):
        print('keys in [{}, {}], values in [{}, {}]'.format(
            min(down2up_edges.keys()),
            max(down2up_edges.keys()),
            min([value for values in down2up_edges.values() for value in values]),
            max([value for values in down2up_edges.values() for value in values])
        ))
    return ls_down2up_edges


def hierarchical_structure_generation(dataset_name, graphs, method, threshold):
    print('is generating hierarchical structure....')
    if method == 'Louvain':
        ls_hierarchical_community = Louvain_community_detection(graphs=graphs)
    elif method == 'GN':
        ls_hierarchical_community = GN_community_detection(graphs=graphs, dataset_name=dataset_name)
    else:
        ls_hierarchical_community = None
    present_community_detection_results(ls_hierarchical_community)

    ls_hierarchical_community = up2down_pipeline(
        graphs=graphs, ls_hierarchical_community=ls_hierarchical_community, threshold=threshold
    )
    ls_up2down_edges, ls_up2down_dicts = up2down_edges(ls_hierarchical_community)
    ls_down2up_edges = down2up_edges(ls_hierarchical_community)

    print('hierarchical structure generation is done')
    return ls_hierarchical_community, ls_up2down_edges, ls_down2up_edges
