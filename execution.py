import time
import numpy as np
import random
import torch

from utils import weights_init, evaluate_results
from preparation import LP_set_up, NC_set_up
from model import Baseline_GNN, HCGNN


def execute_LP(
        args, graphs, features, ls_hierarchical_community,
        ls_adj_same_level, ls_up2down_edges, ls_down2up_edges,
        ls_df_train, ls_df_valid, ls_df_test
):
    ls_data, ls_data_up2down, ls_data_down2up, ls_down2up_torch_arrays, ls_train_user_left, ls_train_user_right, ls_valid_user_left, ls_valid_user_right, ls_test_user_left, ls_test_user_right, ls_train_labels, ls_train_labels_tensor, ls_valid_labels, ls_test_labels = LP_set_up(
        config=args, graphs=graphs,
        features=features,
        ls_hierarchical_community=ls_hierarchical_community,
        ls_adj_same_level=ls_adj_same_level,
        ls_up2down_edges=ls_up2down_edges,
        ls_down2up_edges=ls_down2up_edges,
        ls_df_train=ls_df_train,
        ls_df_valid=ls_df_valid,
        ls_df_test=ls_df_test, device=args.device
    )

    if args.mode == 'baseline':
        model = Baseline_GNN(config=args, graphs=graphs, features=features, ls_labels=None)
    else:
        model = HCGNN(config=args, features=features)
    model = model.to(args.device)
    model.apply(weights_init)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4
    )

    loss_func = torch.nn.BCEWithLogitsLoss()
    out_act = torch.nn.Sigmoid()

    train_results = []
    test_results = []
    valid_results = []

    for epoch_id, epoch in enumerate(range(args.epoch_num)):
        start_epoch = time.time()
        if epoch_id % args.epoch_log == 0:
            print('Epoch {} starts !'.format(epoch_id))
            print('-' * 80)
        total_loss = 0

        for idx, data in enumerate(ls_data):
            data_up2down = ls_data_up2down[idx]
            if args.down2up_gnn == 'MEAN':
                data_down2up = [0]
                down2up_torch_arrays = ls_down2up_torch_arrays[idx]
            else:
                data_down2up = ls_data_down2up[idx]
                down2up_torch_arrays = [0]
            train_user_left = ls_train_user_left[idx]
            train_user_right = ls_train_user_right[idx]
            train_labels_tensor = ls_train_labels_tensor[idx]

            model.train()
            optimizer.zero_grad()

            if args.mode == 'baseline':
                out = model.forward(data)
            else:
                out = model.forward(
                    data=data, data_up2down=data_up2down, data_down2up=data_down2up,
                    down2up_torch_arrays=down2up_torch_arrays
                )

            nodes_left = torch.index_select(out, 0, train_user_left)
            nodes_right = torch.index_select(out, 0, train_user_right)
            preds = torch.sum(nodes_left * nodes_right, dim=-1)
            loss = loss_func(preds, train_labels_tensor).to(args.device)

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.cpu().item()

        # evaluate epoch
        if epoch_id % args.epoch_log == 0:
            model.eval()
            epoch_train_results = []
            epoch_test_results = []
            epoch_valid_results = []
            for idx, data in enumerate(ls_data):
                data_up2down = ls_data_up2down[idx]
                down2up_torch_arrays = ls_down2up_torch_arrays[idx]
                train_user_left = ls_train_user_left[idx]
                train_user_right = ls_train_user_right[idx]
                train_labels = ls_train_labels[idx]
                valid_user_left = ls_valid_user_left[idx]
                valid_user_right = ls_valid_user_right[idx]
                valid_labels = ls_valid_labels[idx]
                test_user_left = ls_test_user_left[idx]
                test_user_right = ls_test_user_right[idx]
                test_labels = ls_test_labels[idx]

                if args.mode == 'baseline':
                    out = model.forward(data)
                else:
                    out = model.forward(
                        data=data, data_up2down=data_up2down, data_down2up=data_down2up,
                        down2up_torch_arrays=down2up_torch_arrays
                    )
                nodes_left_train = torch.index_select(out, 0, train_user_left)
                nodes_right_train = torch.index_select(out, 0, train_user_right)
                pred_train = out_act(torch.sum(nodes_left_train * nodes_right_train, dim=-1))
                pred_train = np.array(out_act(pred_train).view(-1).tolist())
                y_train = np.array(train_labels)

                nodes_left_valid = torch.index_select(out, 0, valid_user_left)
                nodes_right_valid = torch.index_select(out, 0, valid_user_right)
                pred_valid = out_act(torch.sum(nodes_left_valid * nodes_right_valid, dim=-1))
                pred_valid = np.array(out_act(pred_valid).view(-1).tolist())
                y_valid = np.array(valid_labels)

                nodes_left_test = torch.index_select(out, 0, test_user_left)
                nodes_right_test = torch.index_select(out, 0, test_user_right)
                pred_test = out_act(torch.sum(nodes_left_test * nodes_right_test, dim=-1))
                pred_test = np.array(out_act(pred_test).view(-1).tolist())
                y_test = np.array(test_labels)

                epoch_train_results.append(evaluate_results(
                    pred=pred_train, y=y_train, method='roc-auc'
                ))
                epoch_valid_results.append(evaluate_results(
                    pred=pred_valid, y=y_valid, method='roc-auc'
                ))
                epoch_test_results.append(evaluate_results(
                    pred=pred_test, y=y_test, method='roc-auc'
                ))
            print('Evaluating Epoch {}, time {:.3f}, ROC-AUC: Train = {:.4f}, Valid = {:.4f}, Test = {:.4f}'.format(
                epoch_id, time.time() - start_epoch,
                np.mean(epoch_train_results), np.mean(epoch_valid_results), np.mean(epoch_test_results)
            ))
            train_results.append(np.mean(epoch_train_results))
            valid_results.append(np.mean(epoch_valid_results))
            test_results.append(np.mean(epoch_test_results))
            print('Best valid performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.format(
                max(valid_results),
                test_results[valid_results.index(max(valid_results))],
                args.epoch_log * valid_results.index(max(valid_results))
            ))


def execute_NC(
        args, graphs, df_labels, features, ls_hierarchical_community,
        ls_adj_same_level, ls_up2down_edges, ls_down2up_edges
):
    ls_data, features, ls_data_up2down, ls_data_down2up, ls_down2up_torch_arrays, ls_train_nodes, ls_valid_nodes, ls_test_nodes, ls_labels, ls_labels_tensor = NC_set_up(
        config=args, graphs=graphs, df_labels=df_labels, features=features,
        ls_hierarchical_community=ls_hierarchical_community,
        ls_adj_same_level=ls_adj_same_level, ls_up2down_edges=ls_up2down_edges, ls_down2up_edges=ls_down2up_edges,
        device=args.device
    )

    if args.mode == 'baseline':
        model = Baseline_GNN(config=args, graphs=graphs, features=features, ls_labels=None)
    else:
        model = HCGNN(config=args, features=features)
    model = model.to(args.device)
    model.apply(weights_init)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4
    )
    loss_func = torch.nn.NLLLoss()

    train_f1_micro = []
    valid_f1_micro = []
    test_f1_micro = []
    train_f1_macro = []
    valid_f1_macro = []
    test_f1_macro = []
    train_nmi = []
    valid_nmi = []
    test_nmi = []

    for epoch_id, epoch in enumerate(range(args.epoch_num)):
        start_epoch = time.time()
        if epoch_id % args.epoch_log == 0:
            print('Epoch {} starts !'.format(epoch_id))
            print('-' * 80)
        total_loss = 0

        for idx, _ in enumerate(graphs):
            data = ls_data[idx]
            data_up2down = ls_data_up2down[idx]
            if args.down2up_gnn == 'MEAN':
                data_down2up = [0]
                down2up_torch_arrays = ls_down2up_torch_arrays[idx]
            else:
                data_down2up = ls_data_down2up[idx]
                down2up_torch_arrays = [0]
            train_nodes = ls_train_nodes[idx]
            labels_tensor = ls_labels_tensor[idx]

            model.train()
            optimizer.zero_grad()

            if args.mode == 'baseline':
                out = model.forward(data)
            else:
                out = model.forward(
                    data=data, data_up2down=data_up2down, data_down2up=data_down2up,
                    down2up_torch_arrays=down2up_torch_arrays
                )

            pred_train = torch.index_select(out, 0, torch.from_numpy(train_nodes).long().to(args.device))
            loss = loss_func(pred_train, labels_tensor[train_nodes]).to(args.device)

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.cpu().item()

        # evaluate epoch
        model.eval()
        epoch_train_f1_micro = []
        epoch_valid_f1_micro = []
        epoch_test_f1_micro = []
        epoch_train_f1_macro = []
        epoch_valid_f1_macro = []
        epoch_test_f1_macro = []
        epoch_train_nmi = []
        epoch_valid_nmi = []
        epoch_test_nmi = []
        for idx, data in enumerate(ls_data):
            data_up2down = ls_data_up2down[idx]
            if args.down2up_gnn == 'MEAN':
                data_down2up = [0]
                down2up_torch_arrays = ls_down2up_torch_arrays[idx]
            else:
                data_down2up = ls_data_down2up[idx]
                down2up_torch_arrays = [0]
            train_nodes = ls_train_nodes[idx]
            valid_nodes = ls_valid_nodes[idx]
            test_nodes = ls_test_nodes[idx]
            labels = ls_labels[idx]

            if args.mode == 'baseline':
                out = model.forward(data)
            else:
                out = model.forward(
                    data=data, data_up2down=data_up2down, data_down2up=data_down2up,
                    down2up_torch_arrays=down2up_torch_arrays
                )

            pred_train = torch.index_select(out, 0, torch.from_numpy(train_nodes).long().to(args.device))
            pred_valid = torch.index_select(out, 0, torch.from_numpy(valid_nodes).long().to(args.device))
            pred_test = torch.index_select(out, 0, torch.from_numpy(test_nodes).long().to(args.device))

            if epoch_id % args.epoch_log == 0:
                if args.dataset not in ['emails']:
                    epoch_train_f1_micro.append((evaluate_results(
                        pred=pred_train, y=labels, idx=train_nodes, method='mic-f1'
                    )))
                    epoch_valid_f1_micro.append((evaluate_results(
                        pred=pred_valid, y=labels, idx=valid_nodes, method='mic-f1'
                    )))
                    epoch_test_f1_micro.append((evaluate_results(
                        pred=pred_test, y=labels, idx=test_nodes, method='mic-f1'
                    )))
                    epoch_train_f1_macro.append((evaluate_results(
                        pred=pred_train, y=labels, idx=train_nodes, method='mac-f1'
                    )))
                    epoch_valid_f1_macro.append((evaluate_results(
                        pred=pred_valid, y=labels, idx=valid_nodes, method='mac-f1'
                    )))
                    epoch_test_f1_macro.append((evaluate_results(
                        pred=pred_test, y=labels, idx=test_nodes, method='mac-f1'
                    )))
                    print('Evaluating Epoch {}, time {:.3f}'.format(epoch_id, time.time() - start_epoch))
                    print('Micro-f1:  Train = {:.4f}, Valid = {:.4f}, Test Micro-f1 = {:.4f}'.format(
                        np.mean(epoch_train_f1_micro), np.mean(epoch_valid_f1_micro), np.mean(epoch_test_f1_micro)
                    ))
                    print('Macro-f1: Train = {:.4f}, valid = {:.4f}, Macro-f1 = {:.4f}'.format(
                        np.mean(epoch_train_f1_macro), np.mean(epoch_valid_f1_macro), np.mean(epoch_test_f1_macro)
                    ))
                    train_f1_micro.append(np.mean(epoch_train_f1_micro))
                    valid_f1_micro.append(np.mean(epoch_valid_f1_micro))
                    test_f1_micro.append(np.mean(epoch_test_f1_micro))
                    train_f1_macro.append(np.mean(epoch_train_f1_macro))
                    valid_f1_macro.append(np.mean(epoch_valid_f1_macro))
                    test_f1_macro.append(np.mean(epoch_test_f1_macro))
                    print('Best Valid Mic-f1 is {:.4f}, best Test Mic-f1 is {:.4f} and epoch_id is {}'.format(
                        max(valid_f1_micro),
                        test_f1_micro[valid_f1_micro.index(max(valid_f1_micro))],
                        args.epoch_log * valid_f1_micro.index(max(valid_f1_micro))
                    ))
                    print('Best Valid Mac-f1 is {:.4f}, best Test Mac-f1 is {:.4f} and epoch_id is {}'.format(
                        max(valid_f1_macro),
                        test_f1_macro[valid_f1_macro.index(max(valid_f1_macro))],
                        args.epoch_log * valid_f1_macro.index(max(valid_f1_macro))
                    ))
                else:
                    epoch_train_nmi.append(evaluate_results(
                        pred=pred_train, y=labels, idx=train_nodes, method='nmi'
                    ))
                    epoch_valid_nmi.append(evaluate_results(
                        pred=pred_valid, y=labels, idx=valid_nodes, method='nmi'
                    ))
                    epoch_test_nmi.append(evaluate_results(
                        pred=pred_test, y=labels, idx=test_nodes, method='nmi'
                    ))
                    print('NMI: Train = {:.4f},  Valid = {:.4f}, Test NMI = {:.4f}'.format(
                        np.mean(epoch_train_nmi), np.mean(epoch_valid_nmi), np.mean(epoch_test_nmi)
                    ))
                    train_nmi.append(np.mean(epoch_train_nmi))
                    valid_nmi.append(np.mean(epoch_valid_nmi))
                    test_nmi.append(np.mean(epoch_test_nmi))
                    print('Best Valid NMI is {:.4f}, best Test NMI is {:.4f} and epoch_id is {}'.format(
                        max(valid_nmi),
                        test_nmi[valid_nmi.index(max(valid_nmi))], args.epoch_log * valid_nmi.index(max(valid_nmi))
                    ))
