import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import copy
from msr_split1_dataloader import MSR_Dataset_Train
from msr_split1_dataloader import MSR_Dataset_Eval
from youcook2_dataloader import YouCook2_Dataset_Train
from youcook2_dataloader import YouCook2_Dataset_Eval
from vatex_dataloader import VATEX_Dataset_Train
from vatex_dataloader import VATEX_Dataset_Eval
import random

from model_update import FCA
from utils import record_info


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# predefining random initial seeds
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

class ranking_loss(nn.Module):

    def __init__(self, delta):
        super(ranking_loss, self).__init__()
        self.delta = delta

    def forward(self, similarity_matrix):

        # similarity matrix
        similarity_m = similarity_matrix.cuda()

        # correct similarity
        cor_similarity = torch.mul(torch.ones(similarity_m.shape[0], similarity_m.shape[0]).cuda(),
                                   torch.diag(similarity_m)).t().cuda()

        zero_m = torch.zeros(similarity_m.shape[0], similarity_m.shape[0]).cuda()
        loss = torch.max(zero_m, similarity_m - cor_similarity + self.delta) + \
               torch.max(zero_m, similarity_m.t() - cor_similarity + self.delta)

        for i in range(loss.size(0)):
            loss[i][i] = 0

        loss = torch.sum(loss) / similarity_m.shape[0]

        with torch.no_grad():
            recall = []
            score = similarity_m - cor_similarity
            _, pred = score.topk(10, 1, True, True)

            pred = pred.t()  # shape:(10,N)

            target = torch.from_numpy(
                np.arange(0, similarity_m.shape[0])).long().cuda().expand_as(pred)

            correct = pred.eq(target)
            for k in [1, 5, 10]:

                recall.append(torch.sum(correct[:k]))

        return loss, recall


# sequences padding
def pad_sequences(sequences, padding_value=0):

    trailing_dims = sequences[0].shape[1]
    max_len = max([s.shape[0] for s in sequences])
    out_dims = (len(sequences), max_len, trailing_dims)
    out_tensor = torch.from_numpy(sequences[0]).data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]

        out_tensor[i, :length, ...] = torch.from_numpy(tensor)

    return out_tensor


def collate_func(data):
    outs = {}
    batch_len = len(data)
    max_seq_length = max([dic['video_units'].shape[0] for dic in data])
    max_sentence_length = max([dic['phrases'].shape[0] for dic in data])
    sentence_mask_batch = torch.zeros(batch_len, max_sentence_length, dtype=torch.uint8)

    video_mask_batch = torch.zeros(batch_len, max_seq_length, dtype=torch.uint8)

    video_units_batch = []
    sentence_units_batch = []
    for i in range(len(data)):
        dic = data[i]
        video_units_batch.append(dic['video_units'])
        sentence_units_batch.append(dic['phrases'])
        video_mask_batch[i, :dic['video_units'].shape[0]] = 1
        sentence_mask_batch[i, :dic['phrases'].shape[0]] = 1

    outs['video_units'] = pad_sequences(video_units_batch)
    outs['phrases'] = pad_sequences(sentence_units_batch)
    outs['video_mask'] = video_mask_batch
    outs['phrase_mask'] = sentence_mask_batch
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=0.2,
                        help='margin for loss')
    parser.add_argument('--in_features', type=int, default=768,
                        help='gcn input features')
    parser.add_argument('--out_features', type=int, default=768,
                        help='gcn output features')
    parser.add_argument('--gcn_join_dim', type=int, default=1536,
                        help='dimension of joint embedding')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate(1- keep probability)')
    parser.add_argument('--alpha_gcn', type=float, default=0.01,
                        help='Alpha for the leaky_relu')
    parser.add_argument('--n_gcn', type=int, default=2,
                        help='number of gcn layers')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size for a minibatch')
    parser.add_argument('--video_fea_dim', type=int, default=1024,
                        help='the dimension of video feature')
    parser.add_argument('--text_fea_dim', type=int, default=768,
                        help='embedding size for word to vec')
    parser.add_argument('--joint_embedding_dim', type=int, default=768,
                        help='dimension of joint embedding space')
    parser.add_argument('--num_epochs', type=str, default=20,
                        help='number of ephochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for learning')
    parser.add_argument('--max_sentence_length', type=int, default=20,
                        help='maximun length of sentence')
    parser.add_argument('--grad_clip', type=int, default=5,
                        help='grad clip to prevent gradient explode')
    parser.add_argument('--evaluate_every', type=int, default=1000,
                        help='evaluation frequency')
    parser.add_argument('--msrvtt', type=int, default=0,
                        help='on MSRVTT data')
    parser.add_argument('--youcook2', type=int, default=1,
                        help='on YouCook2 data')
    parser.add_argument('--vatex', type=int, default=0,
                        help='on VATEX data')
    parser.add_argument('--save_dir', type=str, default='./model/model_youcook2_update.pth',
                        help='directory to store checkpointed models')
    parser.add_argument('--resume', type=str, default='',
                        help='directory to load checkpointed models')
    parser.add_argument('--evaluate', default=False, action='store_true')
    args = parser.parse_args()
    if os.path.exists('record') is False:
        os.mkdir('record')

    net = FCA(args).to(device)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda')
            net.load_state_dict(checkpoint)
            print("=> loaded checkpoint {} ".format(args.resume))
        else:
            print("=> no checkpoint found at {}".format(args.resume))

    print(net)

    param_num = 0
    for name, para in net.named_parameters():
        shape = para.shape
        num = 1
        for i in range(len(shape)):
            num *= shape[i]
        param_num += num
    print('total prarameters:', param_num)

    if args.msrvtt:
        train_dataset = MSR_Dataset_Train()
        test_dataset = MSR_Dataset_Eval()

    elif args.youcook2:
        train_dataset = YouCook2_Dataset_Train()
        test_dataset = YouCook2_Dataset_Eval()

    else:
        train_dataset = VATEX_Dataset_Train()
        test_dataset = VATEX_Dataset_Eval()

    dataloaders_dict = {
        'train': data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func, num_workers=4, drop_last=True
        ),
        'val': data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func, num_workers=4, drop_last=True
        )

    }

    optimizer = torch.optim.SGD(
       net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.005)

    criterion = ranking_loss(args.delta)
    criterion.cuda()

    bce_loss = nn.BCELoss().cuda()

    net, best_recall = train_model(net, dataloaders_dict, criterion, bce_loss, optimizer, args.num_epochs,
        args.learning_rate)

    torch.save(net.state_dict(), args.save_dir)


def train_model(model, dataloaders, criterion, bce_loss, optimizer, num_epochs, learning_rate):
    since = time.time()
    val_recall1_history = []
    train_recall1_history = []
    val_recall5_history = []
    train_recall5_history = []
    val_recall10_history = []
    train_recall10_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_recall10 = 0.0
    best_recall5 = 0.0
    best_recall1 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        if epoch == 50:
            optimizer.param_groups[0]['lr'] /= 10

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_recall1 = 0.0
            running_recall5 = 0.0
            running_recall10 = 0.0

            for outs_dic in dataloaders[phase]:

                inputs = outs_dic
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    similarity_matrix, A, A_new = model(inputs)

                    bce_loss_sum = 0

                    # optimize reconstructed adjacency matrix A_new
                    for index in range(len(A)):
                        A_index = A[index]
                        A_index = torch.from_numpy(A_index).cuda()
                        A_new_index = A_new[index]
                        bce_loss_sign = bce_loss(A_new_index, A_index)
                        bce_loss_sum = bce_loss_sum + bce_loss_sign

                    classifier_loss, recall = criterion(similarity_matrix)

                    loss = classifier_loss + bce_loss_sum
                    if (phase == 'train'):
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * similarity_matrix.shape[0]
                running_recall1 += recall[0].item()
                running_recall5 += recall[1].item()
                running_recall10 += recall[2].item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_recall1 = 1.0 * running_recall1 / len(dataloaders[phase].dataset)
            epoch_recall5 = 1.0 * running_recall5 / len(dataloaders[phase].dataset)
            epoch_recall10 = 1.0 * running_recall10 / len(dataloaders[phase].dataset)

            info = {'Epoch': [epoch + 1],
                    'learning_rate': [learning_rate],
                    'Loss': [epoch_loss],
                    'epoch_recall@1': [epoch_recall1],
                    'epoch_recall@5': [epoch_recall5],
                    'epoch_recall@10': [epoch_recall10],
                    }
            record_info(info, 'record/MSR' + phase + '.csv')

            if phase == 'val' and epoch_recall10 > best_recall10:
                best_recall10 = epoch_recall10
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_recall10_history.append(epoch_recall10)
                val_recall5_history.append(epoch_recall5)
                val_recall1_history.append(epoch_recall1)
                best_recall5 = max(best_recall5, epoch_recall5)
                best_recall1 = max(best_recall1, epoch_recall1)
            else:
                train_recall10_history.append(epoch_recall10)
                train_recall5_history.append(epoch_recall5)
                train_recall1_history.append(epoch_recall1)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('Best val recall@10 : {:4f} recall@5 : {:4f} recall@1 : {:4f}'. \
            format(best_recall10, best_recall5, best_recall1))
    model.load_state_dict(best_model_wts)
    return model, best_recall10


if __name__ == '__main__':
    main()