# Author: Kun Tian (io.kuntian@gmail.com)
# 2020-02-26

from __future__ import print_function, division

import numpy as np
from sklearn.datasets import fetch_openml

import unsupervised

def sup_flylsh(unsup_fly_lsh, hash_len):

    return


if __name__ == '__main__':

    # load dataset
    mnist = fetch_openml('mnist_784')

    # parameters
    input_dim = 784                                     # dimensionality of input space, d
    max_index = 10000                                   # a mnist subset of size 10,000, N
    sampling_ratio = 0.10                               # raito of PNs that each KC samples from
    num_nn = 200                                        # number of nearest neighbors to compare; 2% of max_index
    num_trials = 5                                      # number of trials; 50 in the paper, 10 in the blog
    hash_lengths = [2, 4, 8, 12, 16, 20, 24, 28, 32]    # hash length, k
    all_mAPs = {}                                       # mean average precision

    # train / test set split
    x = mnist.data
    y = mnist.target
    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
    shuffle_idx = np.random.permutation(60000)
    x_train, y_train = x_train[shuffle_idx], y_train[shuffle_idx]
    x_train = x_train[:max_index]
    y_train = y_train[:max_index]

    # compute LSH and mAP
    for hash_length in hash_lengths:
        embedding_size = int(20 * hash_length)          # dimensionality of projection space, m
        all_mAPs[hash_length] = {}
        all_mAPs[hash_length]['Unsup'] = []
        all_mAPs[hash_length]['Sup'] = []
        for _ in range(num_trials):
            unsup_fly_lsh = unsupervised.flylsh(x_train, hash_length, sampling_ratio, embedding_size)
            unsup_fly_mAP = unsupervised.findmAP(unsup_fly_lsh, num_nn, 1000, x)
            sup_fly_lsh = sup_flylsh(unsup_fly_lsh, hash_length)
            sup_fly_mAP = unsupervised.findmAP(sup_fly_lsh, num_nn, 1000, x)

            print('Unsupervised fly mAP for hash_length %d is %s' % (hash_length, unsup_fly_mAP))
            print('Supervised fly mAP for hash_length %d is %s' % (hash_length, sup_fly_mAP))

            all_mAPs[hash_length]['Unsup'].append(unsup_fly_mAP)
            all_mAPs[hash_length]['Sup'].append(sup_fly_mAP)

    print(all_mAPs)
    print('Done!')

    # plot
    colors = {'Unsup':'blue', 'Sup':'green'}
    unsupervised.plt_mAP(all_mAPs, colors)              # Part II results