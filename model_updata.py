import copy
import time

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
        # GCN
        self.gcn_aten_pool = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.Tanh(),
            nn.Linear(768 // 2, 1),
            nn.Softmax(dim=0)
        )
        

        self.text_pooling = Sentence_Maxpool(self.text_embedding_size, 768)
        
        self.GU_text = Gated_Embedding_Unit(self.text_pooling.out_dim, 768, gating=True)

        # video
      
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
        #max_video_bags = max([outs_dic['video_bags'][i].shape[0] for i in range(batch_size)])
        #max_sentence_bags = max([outs_dic['sentence_bags'][j].shape[0] for j in range(batch_size)])
        #max_length = max_sentence_bags + max_video_bags

        batch_video_feature = []
        batch_sentence_feature = []
        batch_adj = []
        batch_adj_new = []
        for i in range(batch_size):

            video_bags_real = outs_dic['video_bags'][i][outs_dic['video_mask'][i]]
            video_bags_num = video_bags_real.shape[0]
            sentence_bags_real = outs_dic['sentence_bags'][i][outs_dic['sentence_mask'][i]]
            sentence_bags_num = sentence_bags_real.shape[0]
            video_bags_real = self.video_bags_feature(video_bags_real.cuda())
            sentence_bags_features = self.bert(sentence_bags_real.cuda(), output_all_encoded_layers=False)[0]
            sentence_bags_features = self.GU_text(self.text_pooling(sentence_bags_features))
            node_features = torch.cat([video_bags_real, sentence_bags_features], 0)


            # built adj
            video_length = video_bags_real.shape[0]
            
            # initialize A
            A = torch.mm(node_features, torch.t(node_features))
            A = torch.sigmoid(A)
            A = A.detach().cpu().numpy()

            for i in range(A.shape[0]):
                for j in range(i+1, A.shape[0]):

                    A[i][i] = 0
                    A[j][j] = 0

                    if i <= video_length-1 < j:

                        if A[i][j] > 0.5:
                            A[i][j] = 1
                            A[j][i] = 1
                        else:
                            A[i][j] = 0
                            A[j][i] = 0
                    else:
                        A[i][j] = 0
                        A[j][i] = 0
            I = np.eye(A.shape[0])
            A_hat = A + I

            D_hat = np.array(np.sum(A_hat, axis=0))
            D_inv = np.power(D_hat, -0.5)
            D_mat = np.matrix(np.diag(D_inv))
            norm_adj = np.asmatrix(D_mat) @ np.asmatrix(A_hat)
            norm_adj = norm_adj @ np.asmatrix(D_mat)
            norm_adj = torch.from_numpy(norm_adj).float().cuda()
            Z = self.GCN(node_features, norm_adj)
            Z = self.video_bags_feature_all(Z)
            Z = F.dropout(Z, self.dropout, training=self.training)


            # reconstruct A

            A_new_re = torch.mm(Z, torch.t(Z))
            A_new_re = torch.sigmoid(A_new_re)
            A_new = A_new_re.detach().cpu().numpy()
            for i in range(A_new.shape[0]):
                for j in range(i + 1, A_new.shape[0]):

                    A_new[i][i] = 0
                    A_new[j][j] = 0

                    if i <= video_length-1 < j :

                        if A_new[i][j] > 0.4:
                            A_new[i][j] = 1
                            A_new[j][i] = 1
                        else:
                            A_new[i][j] = 0
                            A_new[j][i] = 0
                    else:
                        A_new[i][j] = 0
                        A_new[j][i] = 0


            # select A
            #A_new = A_new > 0.6
            #A_new = A_new.float()
            A_new_hat = A_new + I
            #A_new_hat = A_new.detach().cpu().numpy() + I

            # norm_adj_new = np.asmatrix(D_mat) @ np.asmatrix(A_new_hat)
            # norm_adj_new = norm_adj_new @ np.asmatrix(norm_adj_new)
            # norm_adj_new = torch.from_numpy(norm_adj_new).float().cuda()

            D_new_hat = np.array(np.sum(A_new_hat, axis=0))
            D_new_inv = np.power(D_new_hat, -0.5)
            D_new_mat = np.matrix(np.diag(D_new_inv))
            norm_adj_new = np.asmatrix(D_new_mat) @ np.asmatrix(A_new_hat)
            norm_adj_new = norm_adj_new @ np.asmatrix(D_new_mat)
            norm_adj_new = torch.from_numpy(norm_adj_new).float().cuda()

            Z_new = self.GCN(node_features, norm_adj_new)

            Z_new = self.video_bags_feature_all(Z_new)


            video_feature, sentence_feature = torch.split(Z_new, [video_bags_num, sentence_bags_num])


            # mean -> sum
            local_att_v = self.gcn_aten_pool(video_feature)
            video_feature = torch.sum(video_feature * local_att_v, dim=0)

            local_att_s = self.gcn_aten_pool(sentence_feature)
            sentence_feature = torch.sum(sentence_feature * local_att_s, dim=0)

            #video_feature = torch.mean(video_feature, dim=0, keepdim=False)
            #sentence_feature = torch.mean(sentence_feature, dim=0, keepdim=False)


            batch_adj.append(A)
            batch_adj_new.append(A_new_re)
            batch_video_feature.append(video_feature)
            batch_sentence_feature.append(sentence_feature)

        video_features = torch.stack(batch_video_feature, dim=0)
        sentence_features = torch.stack(batch_sentence_feature, dim=0)

        return sentence_features, video_features, batch_adj, batch_adj_new




if __name__ == '__main__':
    pass