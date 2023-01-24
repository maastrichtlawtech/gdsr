import os, sys, pathlib
import itertools
import argparse
from typing import List, Type

import numpy as np
import pandas as pd
import seaborn as sns

try:
    from src.models.lexical import BM25Retriever
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.models.lexical import BM25Retriever
from src.common.metrics import Metrics
from src.data.text_processor import TextPreprocessor


def run_gridsearch(
        articles: List[str], 
        questions: List[str], 
        ground_truths: List[List[int]], 
        topk: int, 
        outdir: str
    ):
    # Init evaluator and BM25 retriever module.
    evaluator = Metrics(recall_at_k=[100, 200, 500])
    retriever = BM25Retriever(retrieval_corpus=articles, k1=0., b=0.)

    # Create dataframe to store results.
    hyperparameters = ['k1', 'b']
    metrics = [f"recall@{k}" for k in evaluator.recall_at_k]
    grid_df = pd.DataFrame(columns=hyperparameters+metrics)

    # Create all possible combinations of hyperparamaters.
    k1_range = np.arange(0., 8.5, 0.5)
    b_range = np.arange(0., 1.1, 0.1)
    combinations = list(itertools.product(*[k1_range, b_range]))

    # Launch grid search runs.
    for i, (k1, b) in enumerate(combinations):
        print(f"\n\n({i+1}) Model: BM25 - k1={k1}, b={b}")
        retriever.update_params(k1, b)
        retrieved_docs = retriever.search_all(questions, top_k=topk)
        scores = evaluator.compute_all_metrics(ground_truths, retrieved_docs)
        scores.update({**{'k1':k1, 'b':b}, **{f"{metric}@{k}": v for metric, results in scores.items() if isinstance(results, dict) for k,v in results.items()}})
        scores.pop('recall')
        grid_df = grid_df.append(scores, ignore_index=True)
        grid_df.to_csv(os.path.join(outdir, 'bm25_dev_results.csv'), sep=',', float_format='%.5f', index=False)
    return grid_df


def plot_gridsearch_heatmap(results: Type[pd.DataFrame], outdir: str):
    results = results.pivot_table(values='recall@200', index='k1', columns='b')[::-1] *100
    plot = sns.heatmap(
        results,
        annot=True, 
        cmap="YlOrBr", 
        fmt='.1f',
        cbar=False,
        vmin=40, vmax=60,
    )
    plot.get_figure().savefig(os.path.join(outdir, "bm25_dev_heatmap.pdf"))


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    dfA = pd.read_csv(args.articles_path)
    dfQ = pd.read_csv(args.questions_path)
    ground_truths = dfQ['article_ids'].apply(lambda x: list(map(int, x.split(',')))).tolist()

    articles = dfA['article'].values.tolist()
    questions = dfQ['question'].values.tolist()
    if args.do_preprocessing:
        print("Preprocessing queries and documents...")
        cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
        articles = cleaner.preprocess(articles)
        questions = cleaner.preprocess(questions)

    print("Running gridsearch...")
    results = run_gridsearch(
        articles=articles, 
        questions=questions, 
        ground_truths=ground_truths, 
        topk=500, 
        outdir=args.outdir,
    )
    if args.do_plot_heatmap:
        print("Plotting heatmap...")
        plot_gridsearch_heatmap(results=results, outdir=args.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_path", type=str, help="Path of the data file containing the law articles.")
    parser.add_argument("--questions_path", type=str, help="Path of the data file containing the test questions.")
    parser.add_argument("--do_preprocessing", action='store_true', default=False, help="Whether or not to pre-process the articles (lowercasing, lemmatization, and deletion of stopwords, punctuation, and numbers).")
    parser.add_argument("--do_plot_heatmap", action='store_true', default=False, help="Whether or not to plot heatmap of gridsearch results.")
    parser.add_argument("--outdir", type=str, help="Path of the output directory.")
    args, _ = parser.parse_known_args()
    main(args)
