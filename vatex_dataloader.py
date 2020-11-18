from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import random
from pytorch_pretrained_bert import BertTokenizer
import numpy as np
import torch.utils.data as data
from sklearn import metrics
from sklearn.cluster import KMeans


class VATEX_Dataset_Train(data.Dataset):

    def __init__(self):
        super(VATEX_Dataset_Train, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('/data/project/BAG/data/torch-bert-weights/bert-base-uncased-vocab.txt')
        self.video_features = np.load('/media/hing/TXX/bake_bag/data/vatex_data_train.npy', allow_pickle=True)
        self.sentences = np.load('/data/project/video/vatex_sentence/vatex_sentence_train.npy', allow_pickle=True)

        self.len = self.sentences.shape[0]
        print(self.len)

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

        # time ordering of labels
        temp = np.zeros(video_feature.shape[0])
        for frame_idx in range(video_feature.shape[0]):
            temp[frame_idx] = frame_idx / video_feature.shape[0]

        time2label = {}
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label
            r_time = np.mean(temp[total_mask][cluster_mask])
            time2label[r_time] = label

        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]

            kmeans_labels[kmean.labels_ == label] = time_idx

        video_units = []
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label

            video_feature_fragment = video_feature[total_mask][cluster_mask]
            video_units.append(np.mean(video_feature_fragment, axis=0))

        return video_units

    def sentence_phrases_pro(self, sentence_phrases):

        phrases = []
        for i, v in enumerate(sentence_phrases):
            phrases.append(self.get_sentence_ids(sentence_phrases[i]))

        return phrases

    def get_sentence_ids(self, sentence):
        sentence_tokens = self.tokenizer.tokenize(sentence)
        sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
        if len(sentence_ids) >= 15:
            sentence_ids = sentence_ids[:15]
        else:
            for num in range(15 - len(sentence_ids)):
                sentence_ids.append(0)

        return sentence_ids

    def __getitem__(self, index):
        out = {}

        sentences = self.sentences[index]
        rind = random.randint(0, len(sentences) - 1)
        sentence_phrases = sentences[rind]
        phrases = np.array(self.sentence_phrases_pro(sentence_phrases))

        video_feature = self.video_features[index]
        video_units = np.array(self.video_clustering(video_feature))
        out['video_units'] = video_units
        out['phrases'] = phrases

        return out


class VATEX_Dataset_Eval(data.Dataset):

    def __init__(self):
        super(VATEX_Dataset_Eval, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(
            '/data/project/BAG/data/torch-bert-weights/bert-base-uncased-vocab.txt')
        self.video_features = np.load('/media/hing/TXX/bake_bag/data/vatex_data_test.npy', allow_pickle=True)
        self.sentences = np.load('/data/project/video/vatex_sentence/vatex_sentence_test.npy', allow_pickle=True)

        self.len = self.sentences.shape[0]
        print(self.len)

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

        # time ordering of labels
        temp = np.zeros(video_feature.shape[0])
        for frame_idx in range(video_feature.shape[0]):
            temp[frame_idx] = frame_idx / video_feature.shape[0]

        time2label = {}
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label
            r_time = np.mean(temp[total_mask][cluster_mask])
            time2label[r_time] = label

        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]

            kmeans_labels[kmean.labels_ == label] = time_idx

        video_units = []
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label

            video_feature_fragment = video_feature[total_mask][cluster_mask]
            video_units.append(np.mean(video_feature_fragment, axis=0))

        return video_units

    def sentence_phrases_pro(self, sentence_phrases):

        phrases = []
        for i, v in enumerate(sentence_phrases):
            phrases.append(self.get_sentence_ids(sentence_phrases[i]))

        return phrases

    def get_sentence_ids(self, sentence):
        sentence_tokens = self.tokenizer.tokenize(sentence)
        sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
        if len(sentence_ids) >= 15:
            sentence_ids = sentence_ids[:15]
        else:
            for num in range(15 - len(sentence_ids)):
                sentence_ids.append(0)
        return sentence_ids

    def __getitem__(self, index):
        out = {}
        sentences = self.sentences[index]
        rind = random.randint(0, len(sentences) - 1)
        sentence_phrases = sentences[rind]
        phrases = np.array(self.sentence_phrases_pro(sentence_phrases))
        video_feature = self.video_features[index]
        video_units = np.array(self.video_clustering(video_feature))
        out['video_units'] = video_units
        out['phrases'] = phrases

        return out







