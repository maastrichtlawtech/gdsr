import os, sys, pathlib
import json
import argparse
import pandas as pd

try:
    from src.models.lexical import BM25Retriever
except ModuleNotFoundError:
    # Add project root directory to the PYTHONPATH environment variable.
    sys.path.append(str(pathlib.Path().resolve()))
    from src.models.lexical import BM25Retriever
from src.data.text_processor import TextPreprocessor


def main(args):
    print("Loading questions and articles...")
    dfA = pd.read_csv(args.articles_path)
    dfQ = pd.read_csv(args.queries_path)
    question_ids = dfQ['id'].tolist()
    ground_truths = dfQ['article_ids'].astype(str).apply(lambda x: list(map(int, x.split(',')))).tolist()

    print("Preprocessing articles and questions (lemmatizing=True)...")
    cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
    articles = cleaner.preprocess(dfA['article'], lemmatize=True)
    questions = cleaner.preprocess(dfQ['question'], lemmatize=True)


    print("Initializing the BM25 retriever model...")
    retriever = BM25Retriever(retrieval_corpus=articles, k1=1.0, b=0.6)

    print("Running model on questions...")
    retrieved_docs = retriever.search_all(questions, top_k=500)

    print(f"Extracting top-{args.k} negatives...")
    results = dict()
    for q_id, truths_i, preds_i in zip(question_ids, ground_truths, retrieved_docs):
        results[q_id] = [y for y in preds_i if y not in truths_i][:args.k]

    print(f"Saving the results to {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'bm25_negatives.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_path", type=str, help="Path of the data file containing the law articles.")
    parser.add_argument("--queries_path", type=str, help="Path of the data file containing the questions.")
    parser.add_argument("--k", type=int, help="Number of negatives per question.")
    parser.add_argument("--output_dir", type=str, help="Path of the output directory.")
    args, _ = parser.parse_known_args()
    main(args)
