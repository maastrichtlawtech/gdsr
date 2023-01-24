import os, sys, pathlib
import json
import argparse
import pandas as pd
from typing import List

try:
    from src.models.lexical import BM25Retriever
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.models.lexical import BM25Retriever
from src.common.metrics import Metrics
from src.data.text_processor import TextPreprocessor


def run_bm25(
        documents: List[str], 
        queries: List[str], 
        labels: List[List[int]], 
        do_preprocessing: bool,
        k1: int,
        b: int,
        output_dir: str,
    ):
    if do_preprocessing:
        print("Preprocessing queries and documents...")
        cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
        documents = cleaner.preprocess(documents)
        queries = cleaner.preprocess(queries)

    print("Initializing the BM25 retriever...")
    retriever = BM25Retriever(retrieval_corpus=documents, k1=k1, b=b)

    print("Running model on test queries...")
    retrieved_docs = retriever.search_all(queries, top_k=500)

    print("Computing the retrieval scores...")
    evaluator = Metrics(recall_at_k=[100, 200, 500], map_at_k=[100])
    scores = evaluator.compute_all_metrics(labels, retrieved_docs)
    
    print("Saving the scores to {} ...".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'bm25_results.json'), 'w') as f:
        json.dump(scores, f, indent=2)
    return scores


def main(args):
    # Load document, test questions, and ground truth labels.
    dfA = pd.read_csv(args.corpus_path)
    dfQ_test = pd.read_csv(args.questions_path)
    ground_truths = dfQ_test['article_ids'].apply(lambda x: list(map(int, x.split(',')))).tolist()

    # Run BM25.
    scores = run_bm25(
        documents=dfA['article'].values.tolist(), 
        queries=dfQ_test['question'].values.tolist(), 
        labels=ground_truths, 
        do_preprocessing=args.do_preprocessing,
        k1=args.k1, b=args.b,
        output_dir=args.output_dir,
    )
    print(f"Done.\n{scores}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path",  type=str, help="Path of the data file containing the corpus of law articles.")
    parser.add_argument("--questions_path", type=str, help="Path of the data file containing the questions.")
    parser.add_argument("--do_preprocessing", action='store_true', default=False, help="Whether or not to pre-process the articles (lowercasing, lemmatization, and deletion of stopwords, punctuation, and numbers).")
    parser.add_argument("--k1", type=float, help="BM25 hyperparameter.")
    parser.add_argument("--b", type=float, help="BM25 hyperparameter.")
    parser.add_argument("--output_dir", type=str, help="Path of the output directory.")
    args, _ = parser.parse_known_args()
    main(args)
