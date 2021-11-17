# Hierarchical Message-Passing Graph Neural Networks

Implementation of the first implementation model (HC-GNN).

### Required packages
The code has been tested running under Python 3.7.3. with the following packages installed (along with their dependencies):

- numpy == 1.16.5
- pandas == 0.25.1
- scikit-learn == 0.21.2
- networkx == 2.3
- community == 0.13
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

 