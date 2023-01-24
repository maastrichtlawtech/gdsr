#!/bin/bash

K=2.5
B=0.2
TEST_QUERIES_PATH="bsard_v1/questions_fr_test.csv"
CORPUS_PATH="bsard_v1/articles_fr_detailed.csv"
SYNTHETIC_QUERIES_PATH="output/data/synthetic_queries/doc2query_samples.csv"
OUT_PATH="output/baselines/doc2query"

python src/tasks/baselines/doc2query.py \
    --queries_path $TEST_QUERIES_PATH \
    --corpus_path $CORPUS_PATH \
    --synthetic_queries_path $SYNTHETIC_QUERIES_PATH \
    --do_preprocessing \
    --k1 $K \
    --b $B \
    --output_dir $OUT_PATH
