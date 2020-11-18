import os
import argparse
import torch
import torch.utils.data as data

from fca_net.msr_dataloader import MSR_Dataset_Train
from fca_net.msr_dataloader import MSR_Dataset_Eval
from fca_net.model import FCA
from fca_net.utils import compute_metrics, print_computed_metrics


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


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
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='size for a minibatch')
    parser.add_argument('--video_fea_dim', type=int, default=1024,
                        help='the dimension of video feature')
    parser.add_argument('--text_fea_dim', type=int, default=768,
                        help='embedding size for word to vec')
    parser.add_argument('--joint_embedding_dim', type=int, default=768,
                        help='dimension of joint embedding space')
    parser.add_argument('--num_epochs', type=str, default=50,
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
    parser.add_argument('--msrvtt', type=int, default=1,
                        help='on MSRVTT data')
    parser.add_argument('--youcook2', type=int, default=0,
                        help='on YouCook2 data')
    parser.add_argument('--vatex', type=int, default=0,
                        help='on VATEX data')
    parser.add_argument('--save_dir', type=str, default='./model/model_msr.pth',
                        help='directory to store checkpointed models')
    parser.add_argument('--resume', type=str, default='./model/model_msr.pth',
                        help='directory to load checkpointed models')
    parser.add_argument('--evaluate', default=True, action='store_true')
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

        dataloaders_dict = {
            'train': data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func, num_workers=4, drop_last=True
            ),
            'val': data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func, num_workers=4, drop_last=True
            )

        }

    if args.evaluate:
        evaluate(net, dataloaders_dict['val'])
        return


def evaluate(model, dataloaders):
    model.eval()

    with torch.no_grad():
        for outs_dic in dataloaders:
            inputs = outs_dic
            sentence_features, video_features, _, _ = model(inputs)

            feature = torch.matmul(video_features, sentence_features.t())
            m = feature.cpu().detach().numpy()
            metrics = compute_metrics(m)
            print('Video-to-Text:')
            print_computed_metrics(metrics)

            feature2 = torch.matmul(sentence_features, video_features.t())
            n = feature2.cpu().detach().numpy()
            metrics2 = compute_metrics(n)
            print('Text-to-Video:')
            print_computed_metrics(metrics2)


if __name__ == '__main__':
    main()