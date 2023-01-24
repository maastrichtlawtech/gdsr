import math
import itertools
import numpy as np
from tqdm import tqdm
from statistics import mean

from src.common.other import log_step


class TFIDFRetriever:
    def __init__(self, retrieval_corpus):
        self.retrieval_corpus = retrieval_corpus
        self.N = len(retrieval_corpus)
        self.vocab = self._build_vocabulary()
        self.idfs = self._compute_idfs()

    def __repr__(self):
        return f"{self.__class__.__name__}".lower()

    @log_step
    def search_all(self, queries, top_k):
        results = list()
        for q in tqdm(queries, desc='Searching queries'):
            results.append([doc_id for doc_id,_ in self.search(q, top_k)])
        return results

    def search(self, q, top_k):
        results = dict()
        for i, doc in enumerate(self.retrieval_corpus):
            results[i+1] = self.score(q, doc) #NB: '+1' because doc_ids in BSARD start at 1.
        return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def score(self, q, d):
        score = 0.0
        for t in q.split():
            score += self._compute_tfidf(t, d)
        return score

    def _build_vocabulary(self):
        return sorted(set(itertools.chain.from_iterable([doc.lower().split() for doc in self.retrieval_corpus])))

    def _compute_idfs(self):
        idfs = dict.fromkeys(self.vocab, 0)
        for word,_ in idfs.items():
            idfs[word] = self._compute_idf(word)
        return idfs

    def _compute_idf(self, t):
        df = sum([1 if t in doc else 0 for doc in self.retrieval_corpus])
        return math.log10(self.N / (df + 1))

    def _compute_tf(self, t, d):
        return d.split().count(t)

    def _compute_tfidf(self, t, d):
        tf = self._compute_tf(t, d)
        idf = self.idfs[t] if t in self.idfs else math.log10(self.N)
        return tf * idf


class BM25Retriever(TFIDFRetriever):
    def __init__(self, retrieval_corpus, k1, b):
        super().__init__(retrieval_corpus)
        self.k1 = k1
        self.b = b
        self.avgdl = self._compute_avgdl()

    def update_params(self, k1, b):
        self.k1 = k1
        self.b = b
    
    def score(self, q, d):
        score = 0.0
        for t in q.split():
            tf = self._compute_tf(t, d)
            idf = self.idfs[t] if t in self.idfs else math.log10((self.N + 0.5)/0.5)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * len(d.split())/self.avgdl))
        return score

    def _compute_avgdl(self):
        return mean([len(doc.split()) for doc in self.retrieval_corpus])

    def _compute_idf(self, t):
        df = sum([1 if t in doc else 0 for doc in self.retrieval_corpus])
        return math.log10((self.N - df + 0.5) / (df + 0.5))
