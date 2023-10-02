# Hierarchical Message-Passing Graph Neural Networks

Code of the first practical model (HC-GNN).

### Required packages
The code has been tested running under Python 3.7.3. with the following packages installed (along with their dependencies):

- numpy == 1.16.5
- pandas == 0.25.1
- scikit-learn == 0.21.2
- networkx == 2.3
- community (python-louvain) == 0.13
- pytorch == 1.1.0
- torch_geometric == 1.3.2

### Data requirement
All eight datasets we used in the paper are all public datasets which can be downloaded from the internet.

### Code execution
Link prediction:
```
python main.py --task LP --dataset grid --mode basemodel --model HCGNN --layer_num 3 --epoch_num 2001 --lr 0.0001 --relu True --dropout True --drop_ratio 0.5 --same_level_gnn GCN --down2up_gnn MEAN --up2down_gnn GAT --fshot False --SEED 123 --gpu True
```

Node classification and community detection:
```
python main.py --task NC --dataset cora --mode basemodel --model HCGNN --layer_num 2 --epoch_num 201 --lr 0.01 --relu True --dropout False --drop_ratio 0.5 --same_level_gnn GCN --down2up_gnn MEAN --up2down_gnn GCN --fshot True --SEED 1234 --gpu True
``` 

Model hyper-parameters:
```
--task: the target downstream task, "LP; NC; Inductive", type=str, default=LP 
--dataset: dataset name, type=str, default=grid 
--mode: the experiment type, type=str, default=basemodel 
--model: the model name, type=str, default=HCGNN 
--layer_num: the number of layers of primary GNN encoder for within level propagation, type=int, default=3 
--epoch_num: epoch number, type=int, default=2001 
--lr: learning rate, type=float, default=0.0001 
--relu: whether use relu as activation function in the model, type=bool, default=True 
--dropout: whether use dropout component in the model, type=bool, default=True 
--drop_ratio: dropout ratio if use dropout component, type=float, default=0.5 
--same_level_gnn: the GNN encoder for within level propagation, type=str, default=GCN 
--down2up_gnn: define the down2up propagation, type=str, default=MEAN 
--up2down_gnn: define the top2down propagation, type=str, default=GAT 
--fshot: if adopt few-shot learning settings, type=bool, default=False 
--SEED: random seed, type=int, default=123 
--gpu: if use GPU device, type=bool, default=True
```

Two demo file is given to show the execution of link prediction (LP) and node classification (NC) tasks.

## Cite

Please cite our paper if it is helpful in your own work:

```bibtex
@article{ZLP23,
author = {Zhiqiang Zhong and Cheng{-}Te Li and Jun Pang},
title = {Hierarchical Message-Passing Graph Neural Networks},
journal = {Data Mining and Knowledge Discovery (DMKD)},
volume = {37},
number = {1},
pages = {381--408},
publisher = {Springer},
year = {2023},
}
```
