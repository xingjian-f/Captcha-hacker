#coding:utf-8
import numpy as np

def one_hot_encoder(data, whole_set):
    """
     对整个list做encoder，而不是单个record
    """
    ret = []
    for i in data:
        idx = whole_set.index(i)
        ret.append([1 if j==idx else 0 for j in range(len(whole_set))])
    return ret


def one_hot_decoder(data, whole_set):
    ret = []
    for probs in data:
        idx = np.argmax(probs)
        ret.append(whole_set[idx])
    return ret 