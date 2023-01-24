#!/bin/bash

AUGMENT_TYPE=$1

# Data paths.
QUERIES_PATH="bsard_v1/questions_fr_train.csv"
DOCUMENTS_PATH="bsard_v1/articles_fr_detailed.csv"
OUT_PATH="output/data/synthetic_queries/"


if [ "$AUGMENT_TYPE" == "bt" ]; then
    python src/tasks/data_augmentation.py \
        --augmentation $AUGMENT_TYPE \
        --texts_filepath $QUERIES_PATH \
        --source_lang "fr" \
        --target_lang "es" \
        --batch_size 256 \
        --start_id 1109 \
        --output_dir $OUT_PATH

elif [ "$AUGMENT_TYPE" == "qg" ]; then
    python src/tasks/data_augmentation.py \
        --augmentation $AUGMENT_TYPE \
        --texts_filepath $DOCUMENTS_PATH \
        --decoding_strat "topk-nucleus-sampling" \
        --max_query_length 60 \
        --num_gen_samples 5 \
        --batch_size 64 \
        --start_id 1195 \
        --output_dir $OUT_PATH
fi