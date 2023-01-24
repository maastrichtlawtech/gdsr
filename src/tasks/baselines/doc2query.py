import os, sys, pathlib
import json
import argparse
import pandas as pd

try:
    from src.tasks.baselines.bm25 import run_bm25
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.tasks.baselines.bm25 import run_bm25
from src.tasks.data_augmentation import QueryGenerator


def main(args):
    # Load document, test questions, and ground truth labels.
    dfA = pd.read_csv(args.corpus_path)
    dfQ_test = pd.read_csv(args.queries_path)
    ground_truths = dfQ_test['article_ids'].apply(lambda x: list(map(int, x.split(',')))).tolist()

    # Augment the documents with LM generated synthetic queries.
    if not args.synthetic_queries_path:
        generator = QueryGenerator('doc2query/msmarco-french-mt5-base-v1')
        dfQ_syn = generator(
            documents=dfA['article'].values.tolist(),
            document_ids=dfA['id'].values.tolist(),
            decoding_strat="topk-nucleus-sampling",
            max_length=60,
            num_results=5,
            batch_size=64,
            start_id=1109,
            output_dir=args.output_dir,
        )
    else:
        dfQ_syn = pd.read_csv(args.synthetic_queries_path)
    dfQ_syn = dfQ_syn.groupby('article_ids', sort=False, as_index=False)['question'].apply('? '.join)
    dfA['article'] = dfA['article'] + dfQ_syn['question']

    # Run bm25 on augmented articles.
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
    parser.add_argument("--queries_path", type=str, help="Path of the data file containing the questions.")
    parser.add_argument("--synthetic_queries_path", type=str, help="Path of the data file containing the LM generated synthetic questions.")
    parser.add_argument("--do_preprocessing", action='store_true', default=False, help="Whether or not to pre-process the articles (lowercasing, lemmatization, and deletion of stopwords, punctuation, and numbers).")
    parser.add_argument("--k1", type=float, help="BM25 hyperparameter.")
    parser.add_argument("--b", type=float, help="BM25 hyperparameter.")
    parser.add_argument("--output_dir", type=str, help="Path of the output directory.")
    args, _ = parser.parse_known_args()
    main(args)
