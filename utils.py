from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import pandas as pd
import os


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    print(sx, 12)
    d = np.diag(-x)
    print(d, 14)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    print(ind, 19)
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics


def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))


def accuracy(outputs, targets, topk=(1,)):
    # compute the topk accuracy
    maxk = max(topk)
    batch_size = targets.size(0)

    # return the topk scores in every input
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()  # shape:(maxk,N)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def record_info(info, filename):

    print(info)

    df = pd.DataFrame.from_dict(info)
    column_names = ['Epoch', 'learning_rate', 'Loss', 'epoch_recall@1', 'epoch_recall@5', 'epoch_recall@10']
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else:  # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False,
                  index=False, columns=column_names)


def record_best(info, filename):

    print(info)

    df = pd.DataFrame.from_dict(info)
    column_names = ['learning_rate', 'loss_parameter', 'best_val_recall@10']
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else:  # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False,
                  index=False, columns=column_names)



