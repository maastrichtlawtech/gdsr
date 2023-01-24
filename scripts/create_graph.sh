#!/bin/bash

# Data paths.
ARTICLES_PATH="bsard_v1/articles_fr_detailed.csv"
OUT_PATH="output/training/node2vec"

# Build graph of legislation.
python src/utils/bsard_graph.py \
    --create \
    --articles_path $ARTICLES_PATH \
    --output_path $OUT_PATH \ &&

# Draw subgraph from root --node.
python src/utils/bsard_graph.py \
    --plot \
    --nodes_path $OUT_PATH/nodes.txt \
    --edges_path $OUT_PATH/edges.txt \
    --node="La constitution" \
    --output_path $OUT_PATH \ &&
