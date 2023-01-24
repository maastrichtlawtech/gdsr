#!/bin/bash

MODULE_TO_TRAIN=$1 #['biencoder', 'gnn']

#-------------#
# Data paths. #
#-------------#
TRAIN_PATH='bsard_v1/questions_fr_train.csv'
VAL_PATH='bsard_v1/questions_fr_dev.csv' #['None', 'bsard_v1/questions_fr_dev.csv']
TEST_PATH='bsard_v1/questions_fr_test.csv'
SYNTHETIC_PATH='bsard_v1/questions_fr_synthetic.csv' #['None', 'bsard_v1/questions_fr_synthetic.csv']
CORPUS_PATH='bsard_v1/articles_fr_detailed.csv'
NEG_PATH='output/data/hard_negatives/bm25_negatives_original_queries.json' #['output/data/hard_negatives/bm25_negatives_original_queries.json', 'output/data/hard_negatives/bm25_negatives_original_and_doc2query.json']
LOG_PATH='output/training/biencoder/logs/'
CHECK_PATH='output/training/biencoder/checkpoints/'

#-------------------------#
# Hyper-parameters        #
#-------------------------#
Q_CKPT='None'
D_CKPT='None'
MAX_CHUNK_LENGTH=0
MAX_DOC_LENGTH=0
POOL_MODE='None'
ADD_DOC_TITLE='false'
BIENCODER_CKPT='None'
GNN_LAYER_NAME='None'
NUM_GNN_LAYERS=0
NODES_PATH='None'
EDGES_PATH='None'
N2V_VECTORS='None'
if [ "$MODULE_TO_TRAIN" == 'biencoder' ]; then
    # Model.
    Q_CKPT='camembert-base'
    D_CKPT='output/training/domain_adapted_models/LegalCamemBERT'
    MAX_CHUNK_LENGTH=128
    MAX_DOC_LENGTH=1024
    POOL_MODE='max'
    # Training.
    TRAIN_BS=24
    EPOCHS=15
    LR=2.235e-5
    SCHEDULER='linear'
    WARMUP_RATIO=0.05
    WEIGHT_DECAY=0.01
    FP16='true'
    DEEPSPEED='true'
    DS_CONFIG='src/config/ds_zero2.json'
    # +----------------------------------------------------------------------------+
    # | Models                                                   | #params | lang  |
    # |----------------------------------------------------------|---------|-------|
    # | 'camembert-base'                                         | 110M    | fr    |
    # | 'etalab-ia/dpr-{ctx,question}_encoder-fr_qa-camembert'   | 110M    | fr    |s
    # | 'dbmdz/electra-base-french-europeana-cased-discriminator'| 110M    | fr    |
    # | 'bert-base-multilingual-cased'                           | 177M    | multi |
    # | 'xlm-roberta-base'                                       | 278M    | multi |
    # +----------------------------------------------------------------------------+
elif [ "$MODULE_TO_TRAIN" == 'gnn' ]; then
    # Model.
    GNN_LAYER_NAME='gatv2' #['gcn', 'graphsage', 'k-gnn', 'gat', 'gatv2']
    NUM_GNN_LAYERS=3
    BIENCODER_CKPT='output/training/biencoder/checkpoints/2022-10-16_22-52_DSR-LegalCamemBERT-synthetic/last.ckpt/last_fp32.ckpt' #'output/training/biencoder/checkpoints/2022-10-04_03-02_DSR-CamemBERT-synthetic/best_epoch-12_step-63751_val_recall200-0.84.ckpt/best.ckpt' #'output/training/biencoder/checkpoints/2022-10-05_03-31_DSR-CamemBERT-15e-linear/best_epoch-8_step-1700_val_recall200-0.81.ckpt/best.ckpt'
    NODES_PATH='output/data/bsard_graph/bsard_nodes.txt'
    EDGES_PATH='output/data/bsard_graph/bsard_edges.txt'
    # Training.
    TRAIN_BS=512
    EPOCHS=10
    LR=0.0002
    SCHEDULER='linear'
    WARMUP_RATIO=0.0
    WEIGHT_DECAY=0.1
    FP16='false'
    DEEPSPEED='false'
    DS_CONFIG='None'
else
    echo "ERROR: Unknown module to train '$MODULE_TO_TRAIN'. Use either 'biencoder' or 'gnn'."
    exit
fi
#
LOSS_CONFIG='{"type":"cross_entropy","temp":0.01,"metric":"cos"}' #'{"type":"triplet","margin":1.0,"metric":"l2"}'
RATIO_TRAIN_SET=1.0
GRADIENT_CLIP_VAL=1.0
ACCUMULATE_GRAD=1
SEED=42
NUM_WORKERS=4
EVAL_BS=512
#
DO_TRAIN='true'
DO_VAL='true'
DO_TEST='true'
FIND_LR='false'
SAVE_CKPT='true'

# Run training.
python src/tasks/contrastive_training.py \
    --do_train $DO_TRAIN \
    --do_val $DO_VAL \
    --do_test $DO_TEST \
    --ratio_train_set_to_use $RATIO_TRAIN_SET \
    --train_queries_path $TRAIN_PATH \
    --val_queries_path $VAL_PATH \
    --test_queries_path $TEST_PATH \
    --synthetic_queries_path $SYNTHETIC_PATH \
    --documents_path $CORPUS_PATH \
    --hard_negatives_path $NEG_PATH \
    --logs_path $LOG_PATH \
    --checkpoints_path $CHECK_PATH \
    --module_to_train $MODULE_TO_TRAIN \
    --q_model_name_or_path $Q_CKPT \
    --d_model_name_or_path $D_CKPT \
    --max_chunk_length $MAX_CHUNK_LENGTH \
    --max_document_length $MAX_DOC_LENGTH \
    --pooling_mode $POOL_MODE \
    --add_doc_title $ADD_DOC_TITLE \
    --node_vectors_path $N2V_VECTORS \
    --biencoder_ckpt $BIENCODER_CKPT \
    --nodes_path $NODES_PATH \
    --edges_path $EDGES_PATH \
    --gnn_layer_name $GNN_LAYER_NAME \
    --num_gnn_layers $NUM_GNN_LAYERS \
    --loss_config $LOSS_CONFIG \
    --train_batch_size $TRAIN_BS \
    --eval_batch_size $EVAL_BS \
    --epochs $EPOCHS \
    --lr $LR \
    --scheduler $SCHEDULER \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --gradient_clip_val $GRADIENT_CLIP_VAL \
    --accumulate_grad_batches $ACCUMULATE_GRAD \
    --fp16 $FP16 \
    --deepspeed $DEEPSPEED \
    --deepspeed_config_path $DS_CONFIG \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --find_lr $FIND_LR \
    --save_ckpt $SAVE_CKPT \