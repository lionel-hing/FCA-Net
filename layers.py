import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, n_layers, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, input_feature, adj):

        cur_message_layer = input_feature
        embs_messsage = []
        for f in range(self.n_layers):
            n2npool = torch.matmul(adj, cur_message_layer)
            cur_message_layer = torch.matmul(n2npool, self.W)
            cur_message_layer = F.relu(cur_message_layer)
            embs_messsage.append(cur_message_layer)
        embs_message = torch.cat(embs_messsage, dim=1)

        return embs_message


