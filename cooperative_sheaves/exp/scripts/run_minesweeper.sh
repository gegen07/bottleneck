#!/bin/sh

python -m exp.run \
    --dataset=minesweeper \
    --d=3 \
    --layers=5 \
    --gnn_layers=5 \
    --gnn_hidden=64 \
    --pe_size=0 \
    --hidden_channels=32 \
    --epochs=2000 \
    --early_stopping=2000 \
    --left_weights=False \
    --right_weights=True \
    --lr=0.002 \
    --weight_decay=1e-7 \
    --input_dropout=0.2 \
    --dropout=0.2 \
    --use_act=True \
    --use_bias=True \
    --folds=10 \
    --model=CoopSheaf \
    --normalised=True \
    --stop_strategy='acc' \
    --sparse_learner=True \
    --entity="${ENTITY}"