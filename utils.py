import numpy as np
import networkx as nx
from multiprocessing import cpu_count, Pool
import torch
from sklearn.metrics import roc_auc_score, f1_score, normalized_mutual_info_score


def evaluate_results(pred, y, idx=None, method=None):
    if method == 'roc-auc':
        return roc_auc_score(
            y_score=pred, y_true=y
        )
    elif method == 'mic-f1':
        return f1_score(
            y_pred=np.argmax(pred.data.cpu().numpy(), axis=1),
            y_true=np.argmax(y, axis=1)[idx], average='micro'
        )
    elif method == 'mac-f1':
        return f1_score(
            y_pred=np.argmax(pred.data.cpu().numpy(), axis=1),
            y_true=np.argmax(y, axis=1)[idx], average='macro'
        )
    elif method == 'nmi':
        return normalized_mutual_info_score(
            labels_pred=np.argmax(pred.data.cpu().numpy(), axis=1),
            labels_true=np.argmax(y, axis=1)[idx]
        )


def generate_bigram(ls):
    res = []
    for i in range(len(ls)-1):
        res += [(ls[i], item) for item in ls[i+1:]]
    return res


def max_lists(lists):
    return max([item for items in lists for item in items])


def min_lists(lists):
    return min([item for items in lists for item in items])


def max_node(G):
    return max(G)


def min_node(G):
    return min(G)


def single_set_edge_between_community(graph, community, bigrams, threshold):
    res = []
    for bigram in bigrams:
        if sum(1 for _ in nx.algorithms.edge_boundary(
                graph, community['edges_to_lowest'][bigram[0]], community['edges_to_lowest'][bigram[1]]
        )) >= threshold:
            res.append(True)
        else:
            res.append(False)
    return res


def parallel_set_edge_between_community(graph, community, df, threshold):
    # TD: parallel
    # n_core = cpu_count()
    # # n_core = 4
    #
    # bigrams = df['all_bigrams'].values.tolist()
    # pool = Pool(processes=n_core)
    # results = [pool.apply_async(single_set_edge_between_community, args=(
    #     graph, community, bigrams[int(len(bigrams) / cpu_count() * i):int(len(bigrams) / cpu_count() * (i + 1))], threshold)
    # ) for i in range(n_core)]
    # output = [p for res in [result.get() for result in results] for p in res]
    # df['result'] = output

    bigrams = df['all_bigrams'].values.tolist()
    df['result'] = single_set_edge_between_community(
        graph=graph, community=community, bigrams=bigrams, threshold=threshold
    )

    return df


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data = torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
