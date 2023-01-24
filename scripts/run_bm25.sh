#!/bin/bash

K=2.5
B=0.2
QUERIES_PATH="bsard_v1/questions_fr_test.csv"
CORPUS_PATH="bsard_v1/articles_fr_detailed.csv"
OUT_PATH="output/baselines"

python src/tasks/baselines/bm25.py \
    --corpus_path $CORPUS_PATH \
    --questions_path $QUERIES_PATH \
    --do_preprocessing \
    --k1 $K \
    --b $B \
    --output_dir $OUT_PATH
