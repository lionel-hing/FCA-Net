from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import collections

from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch as th
import torch.nn.functional as F
import numpy as np
import torch
import torch.utils.data as data
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.nn.utils.rnn import pack_sequence, pad_sequence


class YouCook2_Dataset_Train(data.Dataset):

    def __init__(self):
        super(YouCook2_Dataset_Train, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('/data/project/BAG/data/torch-bert-weights/bert-base-uncased-vocab.txt')
        train_data = np.load('/data/project/BAG/data/youcook2_all.npz')

        self.video_features = train_data["video_features"][:9548] #(9548, 32, 1024)
        self.sentences = train_data["sentences"][:9548]

        self.len = self.sentences.shape[0]
        print(self.len, 31)

    def __len__(self):

        return self.len
    # 视频特征聚类
    def video_clustering(self, video_feature):
        max_score = 0
        for i in range(2, 6):
            kmean = KMeans(n_clusters=i)
            score = metrics.calinski_harabasz_score(video_feature, kmean.fit_predict(video_feature))

            if score > max_score:
                max_score = score
                K = i
        kmean = KMeans(n_clusters=K)
        kmean.fit(video_feature)
        kmeans_labels = np.asarray(kmean.labels_).copy()
        total_mask = np.ones(video_feature.shape[0], dtype=bool)
        video_feature_bags = []
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label

            video_feature_fragment = video_feature[total_mask][cluster_mask]
            video_feature_bags.append(np.mean(video_feature_fragment, axis=0))

        return video_feature_bags
    # 句子分段
    def sentence_parsing(self, sentence):

        sentence_list = sentence.split()
        sentence_bags = []
        if 'and' in sentence_list:
            sentence_seg = sentence.split('and')

            sentence_bags.append(self.get_sentence_ids(sentence_seg[0]))
            sentence_bags.append(self.get_sentence_ids(sentence_seg[1]))

        else:
            sentence_ids = self.get_sentence_ids(sentence)
            sentence_bags.append(sentence_ids)

        return sentence_bags
    # 固定分好的词组长度为15, 提取对应的词序号
    def get_sentence_ids(self, sentence):
        sentence_tokens = self.tokenizer.tokenize(sentence)
        sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
        if len(sentence_ids) >= 15:
            sentence_ids = sentence_ids[:15]
        else:
            for num in range(15 - len(sentence_ids)):
                sentence_ids.append(0)
        #sentence_ids = np.array(sentence_ids)
        return sentence_ids

    def __getitem__(self, index):
        out = {}
        sentence = self.sentences[index]
        sentence_bags = np.array(self.sentence_parsing(sentence))
        video_feature = self.video_features[index]
        video_bags = np.array(self.video_clustering(video_feature))
        out['video_bags'] = video_bags             # 返回多个分好的视频特征
        out['sentence_bags'] = sentence_bags       # 返回多个分好的词序号

        return out


class YouCook2_Dataset_Eval(data.Dataset):

    def __init__(self):
        super(YouCook2_Dataset_Eval, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(
            '/data/project/BAG/data/torch-bert-weights/bert-base-uncased-vocab.txt')
        train_data = np.load('/data/project/BAG/data/youcook2_all.npz')

        self.video_features = train_data["video_features"][9549:12899] (9548,)
        self.sentences = train_data["sentences"][9549:12899]

        self.len = self.sentences.shape[0]
        print(self.len, 31)

    def __len__(self):

        return self.len

    def video_clustering(self, video_feature):
        max_score = 0
        for i in range(2, 6):
            kmean = KMeans(n_clusters=i)
            score = metrics.calinski_harabasz_score(video_feature, kmean.fit_predict(video_feature))

            if score > max_score:
                max_score = score
                K = i
        kmean = KMeans(n_clusters=K)
        kmean.fit(video_feature)
        kmeans_labels = np.asarray(kmean.labels_).copy()
        total_mask = np.ones(video_feature.shape[0], dtype=bool)
        video_feature_bags = []
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label

            video_feature_fragment = video_feature[total_mask][cluster_mask]
            video_feature_bags.append(np.mean(video_feature_fragment, axis=0))

        return video_feature_bags

    def sentence_parsing(self, sentence):

        sentence_list = sentence.split()
        sentence_bags = []
        if 'and' in sentence_list:
            sentence_seg = sentence.split('and')

            sentence_bags.append(self.get_sentence_ids(sentence_seg[0]))
            sentence_bags.append(self.get_sentence_ids(sentence_seg[1]))

        else:
            sentence_ids = self.get_sentence_ids(sentence)
            sentence_bags.append(sentence_ids)

        return sentence_bags

    def get_sentence_ids(self, sentence):
        sentence_tokens = self.tokenizer.tokenize(sentence)
        sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
        if len(sentence_ids) >= 15:
            sentence_ids = sentence_ids[:15]
        else:
            for num in range(15 - len(sentence_ids)):
                sentence_ids.append(0)
        # sentence_ids = np.array(sentence_ids)
        return sentence_ids

    def __getitem__(self, index):
        out = {}
        sentence = self.sentences[index]
        sentence_bags = np.array(self.sentence_parsing(sentence))
        video_feature = self.video_features[index]
        video_bags = np.array(self.video_clustering(video_feature))
        out['video_bags'] = video_bags
        out['sentence_bags'] = sentence_bags

        return out






if __name__ == '__main__':

    train_dataset = YouCook2_Dataset_Train()


    def pad_sequences(sequences, padding_value=0):

        trailing_dims = sequences[0].shape[1]
        print(trailing_dims)
        max_len = max([s.shape[0] for s in sequences])
        print(max_len, 127)

        out_dims = (len(sequences), max_len, trailing_dims)
        print(torch.from_numpy(sequences[0]).data.new(*out_dims).fill_(padding_value), 128)
        print(out_dims, 130)

        out_tensor = torch.from_numpy(sequences[0]).data.new(*out_dims).fill_(padding_value)
        for i, tensor in enumerate(sequences):
            length = tensor.shape[0]
            print(length, 136)
            # use index notation to prevent duplicate references to the tensor

            out_tensor[i, :length, ...] = torch.from_numpy(tensor)

        return out_tensor


    def collate_func(data):
        outs = {}
        batch_len = len(data)
        max_seq_length = max([dic['video_bags'].shape[0] for dic in data])
        max_sentence_length = max([dic['sentence_bags'].shape[0] for dic in data])
        sentence_mask_batch = torch.zeros(batch_len, max_sentence_length, dtype=torch.uint8)

        video_mask_batch = torch.zeros(batch_len, max_seq_length, dtype=torch.uint8)

        video_bags_batch = []
        sentence_bags_batch = []
        for i in range(len(data)):
            dic = data[i]
            video_bags_batch.append(dic['video_bags'])
            sentence_bags_batch.append(dic['sentence_bags'])
            video_mask_batch[i, :dic['video_bags'].shape[0]] = 1
            sentence_mask_batch[i, :dic['sentence_bags'].shape[0]] = 1

        outs['video_bags'] = pad_sequences(video_bags_batch)
        outs['sentence_bags'] = pad_sequences(sentence_bags_batch)
        outs['video_mask'] = video_mask_batch
        outs['sentence_mask'] = sentence_mask_batch
        return outs

    dataloaders_dict = {
        'train': data.DataLoader(
            train_dataset, batch_size=2, shuffle=False, collate_fn=collate_func, num_workers=0
        )

    }

    for video_feature_bags in dataloaders_dict['train']:


        batch_size = video_feature_bags['video_bags'].shape[0]


        for i in range(batch_size):
            real = video_feature_bags['video_bags'][i][video_feature_bags['video_mask'][i]]
            sentence_real = video_feature_bags['sentence_bags'][i][video_feature_bags['sentence_mask'][i]]
            print(real.shape, 34)
            print(sentence_real.shape, 35)





