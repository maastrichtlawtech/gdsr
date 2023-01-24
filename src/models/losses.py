from enum import Enum
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor as T


class ScoringFunction(Enum):
    """
    This class defines the similarity/distance functions use to compare text embeddings.
    - The distance functions are used in the triplet loss. They take two 2D tensors of (same) size [x, y],
    compute the pairwise distance along dimension y, and returns a 1D tensor of size x.
    - The similarity functions are used for the cross-entropy loss and for retrieval. They take two tensors
    of size [x, y] and [z, y], respectively, compute all possible similarities along dimension y, and returns 
    a 2D tensor of size [x, z].
    """
    COSINE = {
        'distance': (lambda x, y: 1 - F.cosine_similarity(x, y)),
        'similarity': (lambda x, y: torch.mm(F.normalize(x.float()), F.normalize(y.float()).t()))
    }
    DOT = {
        'distance':  (lambda x, y: 100 - torch.mm(x, y.t()).diag()),
        'similarity': (lambda x, y: torch.mm(x.float(), y.float().t()))
    }
    EUCLIDEAN = {
        'distance': (lambda x, y: F.pairwise_distance(x, y, p=2)),
        'similarity': (lambda x, y: 1/(1 + torch.cdist(x.float(), y.float(), p=2)))
    }
    MANHATTAN = {
        'distance': (lambda x, y: F.pairwise_distance(x, y, p=1)),
        'similarity': (lambda x, y: 1/(1 + torch.cdist(x.float(), y.float(), p=1)))
    }


class CrossEntropyLoss(nn.Module):
    """
    This class implements cross-entropy loss. This loss expects as input a batch consisting of sentence pairs 
    (a_1, p_1), (a_2, p_2)..., (a_n, p_n) where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) 
    for i!=j a negative pair. For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 
    1 positive example (p_i) and n-1 negative examples (p_j). It then minimizes the negative log-likehood for 
    softmax normalized scores.
    """
    def __init__(self, scoring_function=ScoringFunction.COSINE, temperature: float = 0.05):
        super(CrossEntropyLoss, self).__init__()
        self.scoring_function = scoring_function
        self.similarity = scoring_function.value['similarity']
        self.temperature = temperature

    def forward(self, *inputs: Tuple[T, T]) -> T:
        rep_anchors, rep_pos_docs, rep_neg_docs = inputs
        rep_docs = torch.cat((rep_pos_docs, rep_neg_docs), 0) if rep_neg_docs is not None else rep_pos_docs #Concat the positive and negative docs (in that order!) if hard negatives are given, otherwise keep the positive docs.
        scores = self.similarity(rep_anchors, rep_docs) / self.temperature
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device) #Tensor[batch_size] where x[i]=i (as q[i] should match with d[i]).
        loss = F.cross_entropy(scores, labels)
        return loss
        

class TripletLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:
    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).
    """
    def __init__(self, scoring_function=ScoringFunction.EUCLIDEAN, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.scoring_function = scoring_function
        self.distance = scoring_function.value['distance']
        self.margin = margin

    def forward(self, *inputs: Tuple[T, T]) -> T:
        rep_anchors, rep_pos_docs, rep_neg_docs = inputs
        assert rep_neg_docs is not None, f"Negatives should be provided when using triplet loss."
        distance_pos = self.distance(rep_anchors, rep_pos_docs)
        distance_neg = self.distance(rep_anchors, rep_neg_docs)
        losses = F.relu(distance_pos - distance_neg + self.margin)
        return losses.mean()
