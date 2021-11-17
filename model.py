import torch

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, GraphUNet
from torch_geometric.utils import dropout_adj


# baseline models
class Baseline_GNN(torch.nn.Module):
    def __init__(self, config, graphs, features, ls_labels):
        super(Baseline_GNN, self).__init__()
        self.config = config
        self.gnn_layers = [512, 256, 128, 64, 32][-(self.config.layer_num+1):]

        self.same_level_gnn_layers = torch.nn.ModuleList()
        if self.config.model == 'GCN':
            if self.config.feature_pre:
                self.linear_pre = torch.nn.Linear(features[0].shape[1], self.gnn_layers[0])
            else:
                self.config.gnn_layers[0] = features[0].shape[1]
            for idx, (in_size, out_size) in enumerate(zip(self.gnn_layers[:-1], self.gnn_layers[1:])):
                self.same_level_gnn_layers.append(GCNConv(in_size, out_size))
        elif self.config.model == 'SAGE':
            if self.config.feature_pre:
                self.linear_pre = torch.nn.Linear(features[0].shape[1], self.gnn_layers[0])
            else:
                self.config.gnn_layers[0] = features[0].shape[1]
            for idx, (in_size, out_size) in enumerate(zip(self.gnn_layers[:-1], self.gnn_layers[1:])):
                self.same_level_gnn_layers.append(SAGEConv(in_size, out_size))
        elif self.config.model == 'GAT':
            if self.config.feature_pre:
                self.linear_pre = torch.nn.Linear(features[0].shape[1], self.gnn_layers[0])
            else:
                self.config.gnn_layers[0] = features[0].shape[1]
            for idx, (in_size, out_size) in enumerate(zip(self.gnn_layers[:-1], self.gnn_layers[1:])):
                self.same_level_gnn_layers.append(GATConv(in_size, out_size))
        elif self.config.model == 'GIN':
            self.same_level_lnn_layers = torch.nn.ModuleList()
            if self.config.feature_pre:
                self.linear_pre = torch.nn.Linear(features[0].shape[1], self.gnn_layers[0])
            else:
                self.config.gnn_layers[0] = features[0].shape[1]
            for idx, (in_size, out_size) in enumerate(zip(self.gnn_layers[:-1], self.gnn_layers[1:])):
                self.same_level_lnn_layers.append(torch.nn.Linear(in_size, out_size))
                self.same_level_gnn_layers.append(GINConv(self.same_level_lnn_layers[idx]))
        elif self.config.model == 'GUNET':
            if graphs[0].number_of_nodes() < 2000:
                pool_ratios = [200 / graphs[0].number_of_nodes(), 0.5]
            else:
                pool_ratios = [2000 / graphs[0].number_of_nodes(), 0.5]
            self.unet = GraphUNet(
                features[0].shape[1], 32, ls_labels[0].shape[1], depth=self.config.layer_num, pool_ratios=pool_ratios
            )

    def forward(self, data):
        if self.config.model == 'GUNET':
            edge_index, _ = dropout_adj(data.edge_index, p=self.config.drop_ratio, 
                                        force_undirected=True,
                                        num_nodes=data.x.shape[0], training=self.training)
            embed = torch.nn.functional.dropout(data.x, p=self.config.drop_ratio, training=self.training)

            embed = self.unet(embed, edge_index)
            
        else:
            x, same_level_edge_index = data.x, data.edge_index

            if self.config.feature_pre:
                embed = self.linear_pre(x)
            else:
                embed = x

            for idx, _ in enumerate(range(len(self.same_level_gnn_layers))):
                if idx != len(self.same_level_gnn_layers)-1:
                    # same level
                    embed = self.same_level_gnn_layers[idx](embed, same_level_edge_index)
                    if self.config.relu:
                        embed = torch.nn.functional.relu(embed) # Note: optional!
                    if self.config.dropout:
                        embed = torch.nn.functional.dropout(embed, p=self.config.drop_ratio, training=self.training)
                else:
                    # same level
                    embed = self.same_level_gnn_layers[idx](embed, same_level_edge_index)

        if self.config.task=='NC':
            embed = torch.nn.functional.log_softmax(embed, dim=1)
        else:
            embed = torch.nn.functional.normalize(embed, p=2, dim=-1)

        return embed


# components of HC-GNN layer
class Down2Up_layer(torch.nn.Module):
    def __init__(self, config, in_size, out_size):
        super(Down2Up_layer, self).__init__()
        self.config = config

        if self.config.down2up_gnn == 'GAT':
            self.nn = GATConv(in_size, out_size)
        elif self.config.down2up_gnn == 'GCN':
            self.nn = GCNConv(in_size, out_size)
        elif self.config.down2up_gnn == 'SAGE':
            self.nn = SAGEConv(in_size, out_size)
        elif self.config.down2up_gnn == 'MEAN':
            self.nn = False
            
    def forward(self, embedding, down2up_paths):
        if type(self.nn) == bool:
            for down2up_array in down2up_paths:
                update_message = torch.mm(down2up_array, embedding)
                embedding = embedding + update_message
                embedding = torch.mul(embedding, 1.0/(down2up_array.sum(-1)+1).unsqueeze(1))
        else:
            embedding = self.nn(embedding, down2up_paths)

        return embedding


class Up2Down_layer(torch.nn.Module):
    def __init__(self, config, in_size, out_size):
        super(Up2Down_layer, self).__init__()
        self.config = config
        if self.config.up2down_gnn == 'GAT':
            self.nn = GATConv(in_size, out_size)
        elif self.config.up2down_gnn == 'GCN':
            self.nn = GCNConv(in_size, out_size)
        elif self.config.up2down_gnn == 'SAGE':
            self.nn = SAGEConv(in_size, out_size)
        
    def forward(self, embedding, up2down_edge_index):
        embedding = self.nn(embedding, up2down_edge_index)
        
        return embedding


class HCGNN_layer(torch.nn.Module):
    def __init__(self, config, in_size, out_size, gnn_type):
        super(HCGNN_layer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.config = config

        self.down2up_layer = Down2Up_layer(self.config, self.in_size, self.in_size)
        if gnn_type == 'GCN':
            self.samle_level_layer = GCNConv(in_size, out_size)
        elif gnn_type == 'SAGE':
            self.samle_level_layer = SAGEConv(in_size, out_size)
        elif gnn_type == 'GAT':
            self.samle_level_layer = GATConv(in_size, out_size)
        elif gnn_type == 'GIN':
            self.same_level_lnn_layer = torch.nn.Linear(in_size, out_size)
            self.samle_level_layer = GINConv(self.same_level_lnn_layer)
        self.up2down_layer = Up2Down_layer(self.config, self.out_size, self.out_size)
        
    def forward(self, embedding, down2up_path, same_level_edge_index, up2down_edge_index):
        # down2up
        embed = self.down2up_layer(embedding=embedding, down2up_paths=down2up_path)
        # same level
        embed = self.samle_level_layer(embed, same_level_edge_index)
        # up2down
        embed = self.up2down_layer(embed, up2down_edge_index)
        
        return embed


# class HCGNN_layer(torch.nn.Module):
#     def __init__(self, in_size, out_size, gnn_type, idx_start, idx_end):
#         super(HCGNN_layer, self).__init__()
#         self.in_size = in_size
#         self.out_size = out_size
#         self.idx_start = idx_start
#         self.idx_end = idx_end
        
#         if self.idx_start:
#             self.down2up_layer = Down2Up_layer(self.config, self.in_size, self.in_size)
        
#         if gnn_type == 'GCN':
#             self.samle_level_layer = GCNConv(self.in_size, self.out_size)
#         elif gnn_type == 'SAGE':
#             self.samle_level_layer = SAGEConv(self.in_size, self.out_size)
#         elif gnn_type == 'GAT':
#             self.samle_level_layer = GATConv(self.in_size, self.out_size)
#         elif gnn_type == 'GIN':
#             self.same_level_lnn_layer = torch.nn.Linear(self.in_size, self.out_size)
#             self.samle_level_layer = GINConv(self.same_level_lnn_layer)
        
#         if self.idx_end:
#             self.up2down_layer = Up2Down_layer(self.config, self.out_size, self.out_size)
        
#     def forward(self, embedding, down2up_path, same_level_edge_index, up2down_edge_index):
#         if self.idx_start:
#             # down2up
#             embedding = self.down2up_layer(embedding=embedding, down2up_paths=down2up_path)
#         # same level
#         embedding = self.samle_level_layer(embedding, same_level_edge_index)
#         if self.idx_end:
#             # up2down
#             embedding = self.up2down_layer(embedding, up2down_edge_index)
        
#         return embedding


# basemodel
class HCGNN(torch.nn.Module):
    def __init__(self, config, features):
        super(HCGNN, self).__init__()
        self.config = config
        self.gnn_layers = [512, 256, 128, 64, 32][-(self.config.layer_num+1):]
        self.cgnn_layers = torch.nn.ModuleList()
        
        if self.config.feature_pre:
            self.linear_pre = torch.nn.Linear(features[0].shape[1], self.gnn_layers[0])
        else:
            self.gnn_layers[0] = features[0].shape[1]

        for idx, (in_size, out_size) in enumerate(zip(self.gnn_layers[:-1], self.gnn_layers[1:])):
            self.cgnn_layers.append(HCGNN_layer(self.config, in_size, out_size, gnn_type=self.config.same_level_gnn))
                
    def forward(self, data, data_up2down, data_down2up, down2up_torch_arrays):
        x, same_level_edge_index = data.x, data.edge_index
        _, up2down_edge_index = data_up2down.x, data_up2down.edge_index
        if self.config.down2up_gnn != 'MEAN':
            _, down2up_edge_index = data_up2down.x, data_down2up.edge_index
        
        if self.config.feature_pre:
            embed = self.linear_pre(x)
        else:
            embed = x
        
        if len(self.cgnn_layers)==1:
            if self.config.down2up_gnn=='MEAN':
                embed = self.cgnn_layers[0](embed, down2up_torch_arrays, same_level_edge_index, up2down_edge_index)
            else:
                embed = self.cgnn_layers[0](embed, down2up_edge_index, same_level_edge_index, up2down_edge_index)
        else:
            for idx in range(len(self.cgnn_layers)):
                if idx != len(self.cgnn_layers)-1:
                    if self.config.down2up_gnn=='MEAN':
                        embed = self.cgnn_layers[idx](
                            embed, down2up_torch_arrays, same_level_edge_index, up2down_edge_index
                        )
                    else:
                        embed = self.cgnn_layers[idx](
                            embed, down2up_edge_index, same_level_edge_index, up2down_edge_index
                        )
                    if self.config.relu:
                        embed = torch.nn.functional.relu(embed) # Note: optional!
                    if self.config.dropout:
                        embed = torch.nn.functional.dropout(embed, p=self.config.drop_ratio, training=self.training)
                else:
                    if self.config.down2up_gnn=='MEAN':
                        embed = self.cgnn_layers[idx](
                            embed, down2up_torch_arrays, same_level_edge_index, up2down_edge_index
                        )
                    else:
                        embed = self.cgnn_layers[idx](
                            embed, down2up_edge_index, same_level_edge_index, up2down_edge_index
                        )

        embed = torch.nn.functional.normalize(embed, p=2, dim=-1)
        if self.config.task == 'NC':
            embed = torch.nn.functional.log_softmax(embed, dim=1)
        return embed
