#!/bin/bash

MODULE_TO_EVAL=$1 #['biencoder', 'gnn']

SET="test" # ["test", "dev"]
QUERIES_PATH="bsard_v1/questions_fr_${SET}.csv"
OUT_NAME="new_${SET}_results"

BIENCODER_CKPT='output/training/biencoder/checkpoints/2022-10-16_22-52_DSR-LegalCamemBERT-synthetic/last.ckpt/last_fp32.ckpt'
GNN_CKPT='output/training/biencoder/checkpoints/2022-10-18_18-24_GATv2-from-DSR-LegalCamemBERT-synthetic-trained-with-synthetic/last.ckpt'

DOCUMENTS_PATH='bsard_v1/articles_fr_detailed.csv'
NODES_PATH='output/data/bsard_graph/bsard_nodes.txt'
EDGES_PATH='output/data/bsard_graph/bsard_edges.txt'
EVAL_BS=512
SEED=42
NUM_WORKERS=4

python src/tasks/evaluation.py \
    --queries_path $QUERIES_PATH \
    --documents_path $DOCUMENTS_PATH \
    --nodes_path $NODES_PATH \
    --edges_path $EDGES_PATH \
    --module_to_test $MODULE_TO_EVAL \
    --biencoder_ckpt_path $BIENCODER_CKPT \
    --gnn_ckpt_path $GNN_CKPT \
    --eval_batch_size $EVAL_BS \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --out_filename $OUT_NAME
