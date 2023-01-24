#!/bin/bash

PRETRAINED_CKPT="camembert-base"
NEW_MODEL_NAME="legalbert-fr"
EPOCHS=200
BATCH_SIZE=32
FP16='true'
CORPUS_PATH="bsard_v1/articles_fr.csv"
OUT_DIR="output/training/domain_adapted_models"

# Create the MLM training corpus (.txt file with one documment per line).
if [ ! -f "$OUT_DIR/articles_fr.txt" ]; then
    python -c 'from src.common.other import corpus2txt; corpus2txt("'$CORPUS_PATH'", "'$OUT_DIR'")'
fi

# Run MLM training.
python src/tasks/mlm_training.py \
    --model_name_or_path $PRETRAINED_CKPT \
    --train_file $OUT_DIR/articles_fr.txt \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --fp16 $FP16 \
    --output_dir $OUT_DIR/$NEW_MODEL_NAME/ \
    --do_train 'true' \
    --warmup_ratio 0.05 \
    --adam_epsilon 1e-7 \
    --logging_steps 250 \
    --overwrite_output_dir 'true'