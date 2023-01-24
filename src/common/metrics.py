from typing import List
from collections import defaultdict

import numpy as np
from statistics import mean


class Metrics:
    def __init__(self, recall_at_k: List[int], map_at_k: List[int] = [], mrr_at_k: List[int] = []):
        self.recall_at_k = recall_at_k
        self.map_at_k = map_at_k
        self.mrr_at_k = mrr_at_k

    def compute_all_metrics(self, all_ground_truths, all_results):
        scores = defaultdict(dict)
        for k in self.recall_at_k:
            scores['recall'][k] = self.compute_mean_score(self.recall, all_ground_truths, all_results, k)
        for k in self.map_at_k:
            scores['map'][k] = self.compute_mean_score(self.average_precision, all_ground_truths, all_results, k)
        for k in self.mrr_at_k:
            scores['mrr'][k] = self.compute_mean_score(self.reciprocal_rank, all_ground_truths, all_results, k)
        scores['r-precision'] = self.compute_mean_score(self.r_precision, all_ground_truths, all_results)
        return scores

    def compute_mean_score(self, score_func, all_ground_truths: List[List[int]], all_results: List[List[int]],  k: int = None):
        return mean([score_func(truths, res, k) for truths, res in zip(all_ground_truths, all_results)])

    def average_precision(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        p_at_k = [self.precision(ground_truths, results, k=i+1) if d in ground_truths else 0 for i, d in enumerate(results[:k])]
        return sum(p_at_k)/len(ground_truths)

    def reciprocal_rank(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        return max([1/(i+1) if d in ground_truths else 0.0 for i, d in enumerate(results[:k])])

    def r_precision(self, ground_truths: List[int], results: List[int], R: int = None):
        R = len(ground_truths)
        relevances = [1 if d in ground_truths else 0 for d in results[:R]]
        return sum(relevances)/R

    def recall(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(relevances)/len(ground_truths)

    def precision(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(relevances)/len(results[:k])

    def fscore(self, ground_truths: List[int], results: List[int], k: int = None):
        p = self.precision(ground_truths, results, k)
        r = self.recall(ground_truths, results, k)
        return (2*p*r)/(p+r) if (p != 0.0 or r != 0.0) else 0.0
