# Author: Kun Tian (io.kuntian@gmail.com)
# 2020-02-26
# Reference: https://github.com/dataplayer12/Fly-LSH

from __future__ import print_function, division

import numpy as np
from sklearn.datasets import fetch_openml
from bokeh.plotting import figure, show

def flylsh(data, hash_len, sampling_r, embed_size):
    '''
    Fly-LSH
    :return: Fly-LSH
    '''

    # Step 1: Divisive normalization
    data = data - np.mean(data, axis=1)[:, None]

    # Step 2: Random projection
    num_projections = int(sampling_r * data.shape[1])         # number of projections from PNs to KCs
    weights = np.random.random((data.shape[1], embed_size))
    yindices = np.arange(weights.shape[1])[None, :]
    xindices = weights.argsort(axis=0)[-num_projections:, :]
    weights = np.zeros_like(weights, dtype=np.bool)
    weights[xindices, yindices] = True                        # sparse projection vectors

    # Step 3: Hashing by winner-take-all
    all_activations = np.dot(data, weights)
    xindices = np.arange(data.shape[0])[:, None]
    yindices = all_activations.argsort(axis=1)[:, -hash_len:]
    hashes = np.zeros_like(all_activations, dtype=np.bool)
    hashes[xindices, yindices] = True                         # choose topk activations

    return hashes

def convlsh(data, hash_len):
    '''
    Conventional LSH 
    :return: LSH
    '''

    data = data - np.mean(data, axis=1)[:, None]    # divisive normalization
    weights = np.random.random((data.shape[1], hash_len))
    hashes = np.dot(data, weights) > 0

    return hashes

def query(qidx, n_nn, fly_lsh):
    '''
    :return: predicted nearest neighbors in m-dimensional hash space
    '''
    L1_distances = np.sum(np.abs(fly_lsh[qidx, :] ^ fly_lsh), axis=1)
    n_nn = min(fly_lsh.shape[0], n_nn)
    NNs = L1_distances.argsort()
    NNs = NNs[(NNs != qidx)][:n_nn]

    return NNs

def true_nns(qidx, n_nn, data):    # Q: should this data be x or divisive normalized data?
    sample = data[qidx, :]
    tnns = np.sum((data - sample) ** 2, axis=1).argsort()[:n_nn + 1]
    tnns = tnns[(tnns != qidx)]
    if n_nn < data.shape[0]:
        assert len(tnns) == n_nn, 'n_nn={}'.format(n_nn)
    return tnns


def construct_true_nns(sample_idx, n_nn, data):
    '''
    :return: true nearest neighbors in input space
    '''
    all_NNs = np.zeros((len(sample_idx), n_nn))
    for idx1, idx2 in enumerate(sample_idx):
        all_NNs[idx1, :] = true_nns(idx2, n_nn, data)
    return all_NNs

def AP(predictions, truth, fly_lsh):
    ''' 
    :return: average precision
    '''
    assert len(predictions) == len(truth) or len(predictions) == fly_lsh.shape[0]
    precisions = [len((set(predictions[:idx]).intersection(set(truth[:idx])))) / idx for \
                  idx in range(1, len(truth) + 1)]
    return np.mean(precisions)

def findmAP(fly_lsh, n_nn, n_samples, data):
    '''
    :param lsh: locality-sensitive hash
    :param n_nn: number of nearest neighbors to compare
    :param n_samples: 1000
    :param data: MNIST data
    :return: mean average precision
    '''
    sample_indices = np.random.choice(data.shape[0], n_samples)
    all_NNs = construct_true_nns(sample_indices, n_nn, data)
    allAPs = []

    for eidx, didx in enumerate(sample_indices):        # eidx: enumeration id; didx: idx of sample in data
        this_nns = query(didx, n_nn, fly_lsh)
        this_AP = AP(list(this_nns),list(all_NNs[eidx,:]), fly_lsh)
        allAPs.append(this_AP)

    return np.mean(allAPs)

def plt_mAP(all_results, colors, hash_lengths=None, keys=None):
    if hash_lengths is None:
        hash_lengths = sorted(all_results.keys())

    if keys is None:
        keys = list(all_results[hash_lengths[0]].keys())

    Lk = len(keys)

    # compute mean and std of mAPs
    curve_ylabel = 'mean Average Precision (mAP)'
    min_y = 0
    mean = lambda x, n: np.mean(all_results[x][n])
    stdev = lambda x, n: np.std(all_results[x][n])

    p = figure(x_range=[str(h) for h in hash_lengths], title='Fly-LSH vs. Conventional LSH on MNIST')
    delta = 0.5 / (Lk + 1)
    deltas = [delta * i for i in range(-Lk, Lk)][1::2]
    assert len(deltas) == Lk, 'Bad luck'

    x_axes = np.sort(np.array([[x - 0.5 + d for d in deltas] for x in range(1, 1 + len(hash_lengths))]), axis=None)
    means = [mean(hashl, name) for name, hashl in zip(keys * len(hash_lengths), sorted(hash_lengths * Lk))]
    stds = [stdev(hashl, name) for name, hashl in zip(keys * len(hash_lengths), sorted(hash_lengths * Lk))]

    for i in range(len(hash_lengths)):
        for j in range(Lk):
            p.vbar(x=x_axes[Lk * i + j], width=delta, bottom=0, top=means[Lk * i + j], color=colors[keys[j]],
                   legend=keys[j])

    err_xs = [[i, i] for i in x_axes]
    err_ys = [[m - s, m + s] for m, s in zip(means, stds)]
    p.y_range.bounds = (min_y, np.floor(10 * max(means)) / 10 + 0.1)
    p.multi_line(err_xs, err_ys, line_width=2, color='black')
    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    p.xaxis.axis_label = 'Hash length (k)'
    p.yaxis.axis_label = curve_ylabel
    show(p)

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

    x = mnist.data[:max_index]
    y = mnist.target[:max_index]

    # compute LSH and mAP
    for hash_length in hash_lengths:
        embedding_size = int(20 * hash_length)          # dimensionality of projection space, m
        all_mAPs[hash_length] = {}
        all_mAPs[hash_length]['Fly'] = []
        all_mAPs[hash_length]['Conv'] = []
        for _ in range(num_trials):
            fly_lsh = flylsh(x, hash_length, sampling_ratio, embedding_size)
            fly_mAP = findmAP(fly_lsh, num_nn, 1000, x)
            conv_lsh = convlsh(x, hash_length)
            conv_mAP = findmAP(conv_lsh, num_nn, 1000, x)

            print('fly mAP for hash_length %d is %s' % (hash_length, fly_mAP))
            print('conv mAP for hash_length %d is %s' % (hash_length, conv_mAP))

            all_mAPs[hash_length]['Fly'].append(fly_mAP)
            all_mAPs[hash_length]['Conv'].append(conv_mAP)

    print(all_mAPs)
    print('Done!')

    # plot
    colors = {'Fly':'blue', 'Conv':'red'}
    plt_mAP(all_mAPs, colors)                           # Part I results

