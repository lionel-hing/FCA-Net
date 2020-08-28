import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from fvt.layers import GraphConvolution, GraphAttentionLayer
from scipy.sparse import csr_matrix
import scipy.sparse as sp

from pytorch_pretrained_bert import BertModel, BertTokenizer
from numpy import linalg

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


class joint_embedding(nn.Module):
    def __init__(self, ori_fea_dim, joint_embedding_dim):
        super(joint_embedding, self).__init__()
        self.img_feature = nn.Linear(ori_fea_dim, joint_embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, img):
        img_feature = self.img_feature(img)
        img_feature = self.tanh(img_feature)  # [batch_size,joint_embedding_dim]
        return img_feature


class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, gating=True):
        super(Gated_Embedding_Unit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)
        self.gating = gating

    def forward(self, x):
        x = self.fc(x)
        if self.gating:
            x = self.cg(x)
        x = F.normalize(x)
        return x


class Context_Gating(nn.Module):
    def __init__(self, dimension, add_batch_norm=False):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)


class Sentence_Maxpool(nn.Module):
    def __init__(self, word_dimension, output_dim, relu=True):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)
        self.out_dim = output_dim
        self.relu = relu

    def forward(self, x):
        x = self.fc(x)
        if self.relu:
            x = F.relu(x)
        return torch.max(x, dim=1)[0]


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.init_scale = args.init_scale

        # regularization strength
        self.alpha = args.alpha
        self.gcn_input_dim = args.in_features
        self.gcn_output_dim = args.out_features
        self.gcn_join_dim = args.gcn_join_dim
        self.dropout = args.dropout

        self.GCN = GraphConvolution(self.gcn_input_dim, self.gcn_output_dim, args.n_gcn)

    def forward(self, node_features, adj):
        Z = self.GCN(node_features, adj)

        return Z


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.init_scale = args.init_scale

        # regularization strength
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.kee_prob = args.keep_prob
        self.video_fea_dim = args.video_fea_dim
        self.joint_embedding_dim = args.joint_embedding_dim
        self.text_embedding_size = args.text_embedding_size
        self.hidden_size = args.hidden_size
        self.graph_nodes = args.graph_nodes
        self.gcn_input_dim = args.in_features
        self.gcn_output_dim = args.out_features
        self.gcn_join_dim = args.gcn_join_dim
        self.dropout = args.dropout


        # BERT
        self.bert = BertModel.from_pretrained('/data/project/GGBT/data/torch-bert-weights/bert-base-uncased/bert-base-uncased')

        self.GCN = GCN(args)

        self.gcn_aten_pool = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.Tanh(),
            nn.Linear(768 // 2, 1),
            nn.Softmax(dim=0)
        )

        self.text_pooling = Sentence_Maxpool(self.text_embedding_size, 768)
        #self.Gu_graph = Gated_Embedding_Unit(self.graph_pooling.out_dim, self.hidden_size, gating=True)

        self.GU_text = Gated_Embedding_Unit(self.text_pooling.out_dim, 768, gating=True)
        self.video_bags_feature = joint_embedding(1024, 768)
        self.video_bags_feature_all = Gated_Embedding_Unit(1536, 768)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):

        outs_dic = inputs
        batch_size = outs_dic['video_bags'].shape[0]

        batch_video_feature = []
        batch_sentence_feature = []
        batch_adj = []
        batch_adj_new = []
        # 对每个batch中的视频特征个数 和文本个数做处理
        for i in range(batch_size):
            # get 真实视频特征的个数
            video_bags_real = outs_dic['video_bags'][i][outs_dic['video_mask'][i]]
            video_bags_num = video_bags_real.shape[0]
            # get 真实文本特征的个数
            sentence_bags_real = outs_dic['sentence_bags'][i][outs_dic['sentence_mask'][i]]
            sentence_bags_num = sentence_bags_real.shape[0]
            video_bags_real = self.video_bags_feature(video_bags_real.cuda()) # (x, 1024)-> (x, 768) x: 视频特征个数
            sentence_bags_features = self.bert(sentence_bags_real.cuda(), output_all_encoded_layers=False)[0] #(y, 15, 768)
            sentence_bags_features = self.GU_text(self.text_pooling(sentence_bags_features)) # 获取最终文本特征 (y, 768) y: 文本特征个数
            node_features = torch.cat([video_bags_real, sentence_bags_features], 0) # (x+y, 768)


            # 建立邻接矩阵
            length = video_bags_num + sentence_bags_num
            A = np.zeros((length, length), dtype=np.float32)
            for i in range(length):
                for j in range(i+1, length):
                    if (i <= video_bags_num-1 and j > video_bags_num-1):

                        if torch.cosine_similarity(node_features[i], node_features[j], dim=0) > 0.01:
                            A[i][j] = 1
                            A[j][i] = 1
            # 传入GCN前,对邻接矩阵做正则化.
            I = np.eye(A.shape[0])
            A_hat = A + I

            D_hat = np.array(np.sum(A_hat, axis=0))
            D_inv = np.power(D_hat, -0.5)
            D_mat = np.matrix(np.diag(D_inv))
            norm_adj = np.asmatrix(D_mat) @ np.asmatrix(A_hat)
            norm_adj = norm_adj @ np.asmatrix(D_mat)
            norm_adj = torch.from_numpy(norm_adj).float().cuda()
            # get 所有节点特征
            Z = self.GCN(node_features, norm_adj)
            Z = self.video_bags_feature_all(Z)

            Z = F.dropout(Z, self.dropout, training=self.training)

            # reconstruct A
            A_new = torch.mm(Z, torch.t(Z))
            A_new = torch.sigmoid(A_new)

            # select A
            #A_new = A_new > 0.6
            #A_new = A_new.float()
            # 利用A_new 再次get 所有节点特征 Z_new
            A_new_hat = A_new.detach().cpu().numpy() + I
            D_new_hat = np.array(np.sum(A_new_hat, axis=0))
            D_new_inv = np.power(D_new_hat, -0.5)
            D_new_mat = np.matrix(np.diag(D_new_inv))
            norm_adj_new = np.asmatrix(D_new_mat) @ np.asmatrix(A_new_hat)
            norm_adj_new = norm_adj_new @ np.asmatrix(D_new_mat)
            norm_adj_new = torch.from_numpy(norm_adj_new).float().cuda()

            Z_new = self.GCN(node_features, norm_adj_new)
            Z_new = self.video_bags_feature_all(Z_new)
            # get 优化后的视频特征和文本特征
            video_feature, sentence_feature = torch.split(Z_new, [video_bags_num, sentence_bags_num]) # (x, 768) # (y, 768)

            # mean -> sum
            #
            local_att_v = self.gcn_aten_pool(video_feature)
            video_feature = torch.sum(video_feature * local_att_v, dim=0) #(768)

            local_att_s = self.gcn_aten_pool(sentence_feature)
            sentence_feature = torch.sum(sentence_feature * local_att_s, dim=0) #(768)

            #video_feature = torch.mean(video_feature, dim=0, keepdim=False)
            #sentence_feature = torch.mean(sentence_feature, dim=0, keepdim=False)
            #重新将数据组成 batch

            batch_adj.append(A)
            batch_adj_new.append(A_new)
            batch_video_feature.append(video_feature)
            batch_sentence_feature.append(sentence_feature)

        video_features = torch.stack(batch_video_feature, dim=0)  # (128, 768)
        sentence_features = torch.stack(batch_sentence_feature, dim=0)  # (128, 768)

        return sentence_features, video_features, batch_adj, batch_adj_new




if __name__ == '__main__':
    pass