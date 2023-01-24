#!/bin/bash

# Data paths.
QUERIES_PATH="output/data/synthetic_queries/doc2query_samples.csv" #["bsard_v1/questions_fr_train.csv", "output/data/synthetic_queries/backtranslations_fr-es.csv", "output/data/synthetic_queries/doc2query_samples.csv"]
ARTICLES_PATH="bsard_v1/articles_fr.csv"
OUT_PATH="output/data/hard_negatives/"
NUM_NEG_PER_QUERY=10


python src/tasks/bm25_negatives.py \
    --queries_path $QUERIES_PATH \
    --articles_path $ARTICLES_PATH \
    --k $NUM_NEG_PER_QUERY \
    --output_dir $OUT_PATH
