#!/bin/bash

python train_decoder.py \
    --lr 1e-5 \
    --num_epochs 25 \
    --trained_path ./trained_models/COCO/ \
    --backbone ViT-B/32 \
    --device 2 \
    --feat_batch_size 256 \
    --batch_size 128
