import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, n_layers, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, input_feature, adj):
        #lv = 0
        cur_message_layer = input_feature
        embs_messsage = []
        for f in range(self.n_layers):
            n2npool = torch.matmul(adj, cur_message_layer)
            cur_message_layer = torch.matmul(n2npool, self.W)

            embs_messsage.append(cur_message_layer)
        embs_message = torch.cat(embs_messsage, dim=1)
        # print(side_message.shape, 67)
        # message_out = torch.mean(side_message, dim=1)
        # print(message_out.shape, 68)
        return embs_message


# class GraphConvolution(nn.Module):
#
#     def __init__(self, in_features, out_features, n_layers, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.n_layers = n_layers
#
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#
#     def forward(self, input_feature, adj):
#         lv = 0
#         cur_message_layer = input_feature
#
#         while lv < self.n_layers:
#             n2npool = torch.matmul(adj, cur_message_layer)
#             cur_message_layer = torch.matmul(n2npool, self.W)
#             lv += 1
#         return cur_message_layer

