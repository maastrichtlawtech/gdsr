#!/bin/bash

QUERIES_PATH="bsard_v1/questions_fr_dev.csv"
CORPUS_PATH="bsard_v1/articles_fr_detailed.csv"
OUT_PATH="output/baselines"

python src/tasks/baselines/bm25_tuning.py \
    --articles_path $CORPUS_PATH \
    --questions_path $QUERIES_PATH \
    --do_preprocessing \
    --do_plot_heatmap \
    --outdir $OUT_PATH
