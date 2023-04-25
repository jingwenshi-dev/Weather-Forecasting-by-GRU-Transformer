import numpy as np
import torch
from torch import nn
import collections
import math
torch.manual_seed(42)


def app_sigma(size):
    # using equation given in  getGaussianKernel() on page:
    # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    return 0.3*((size-1)*0.5 - 1) + 0.8


def gaussian_2d(size, sigma=None):
    # using equation given in  getGaussianKernel() on page:
    # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    a = []
    if sigma is None:
        sigma = app_sigma(size)
    down = 2 * sigma ** 2
    for i in range(size):
        up = -(i - (size - 1)/2) ** 2
        a.append(np.exp(up/down))
    gaussian_1d = (np.array(a)/np.sum(np.array(a))).reshape(size, 1)
    return np.multiply(gaussian_1d.T, gaussian_1d)


# helper function by towaki
def normalized_grid(width, height):
    """Returns grid[x,y] -> coordinates for a normalized window.

    Args:
        width, height (int): grid resolution
    """

    # These are normalized coordinates
    # i.e. equivalent to 2.0 * (fragCoord / iResolution.xy) - 1.0
    window_x = np.linspace(-1, 1, num=width) * (width / height)
    window_x += np.random.rand(*window_x.shape) * (1. / width)
    window_y = np.linspace(1, -1, num=height)
    window_y += np.random.rand(*window_y.shape) * (1. / height)
    coord = np.array(np.meshgrid(window_x, window_y, indexing='xy')).transpose(2, 1, 0)

    return coord


def my_normalized_grid(width, height, start, end):
    x_dim = torch.linspace(start, end, width)
    y_dim = torch.linspace(start, end, height)
    meshx, meshy = torch.meshgrid((x_dim, y_dim))
    return torch.stack((meshx, meshy), 2)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # pred shape (batch_size,num_steps,vocab_size)
    # label shape(batch_size,num_steps)
    # valid_len shape(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def vae_loss(pred, label, valid_len, mu, logvar):
    CE_loss = MaskedSoftmaxCELoss()
    CE = CE_loss(pred, label, valid_len)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return CE + KLD


def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    sum_list = []
    # for p in params:
    #     if p.grad is not None:
    #         sum_list.append(torch.sum(p.grad ** 2))
    # norm = torch.sqrt(sum(sum_list))
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params if p.grad is not None))
    if norm > theta:
        for param in params:
            if param.grad is not None:
                param.grad[:] *= theta / norm


def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def get_sent_len(file):
    sent_len = []
    with open(file, 'r', encoding='UTF-8') as f:
        for line in f:
            sent_len.append(len(line.split()))
    return sent_len


# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def corpus_bleu(hypotheses, references):
    refs = []
    hyps = []
    count = 0
    total_score = 0.0

    assert (sorted(hypotheses.keys()) == sorted(references.keys()))
    Ids = list(hypotheses.keys())
    ind_score = dict()

    for id in Ids:
        hyp = hypotheses[id][0].split()
        ref = [r.split() for r in references[id]]
        hyps.append(hyp)
        refs.append(ref)

        score = compute_bleu([ref], [hyp], smooth=True)[0]
        total_score += score
        count += 1
        ind_score[id] = score

    avg_score = total_score / count
    corpus_bleu = compute_bleu(refs, hyps, smooth=True)[0]
    return corpus_bleu, avg_score, ind_score


