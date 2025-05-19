#!/bin/sh

python -m exp.run \
    --dataset=questions \
    --d=5 \
    --layers=4 \
    --gnn_layers=4 \
    --gnn_hidden=32 \
    --pe_size=0 \
    --hidden_channels=64 \
    --epochs=2000 \
    --early_stopping=200 \
    --left_weights=False \
    --right_weights=False \
    --lr=0.002 \
    --weight_decay=1e-7 \
    --input_dropout=0.2 \
    --dropout=0.2 \
    --use_act=True \
    --use_bias=True \
    --folds=10 \
    --model=CoopSheaf \
    --normalised=True \
    --sparse_learner=True \
    --stop_strategy='acc' \
    --entity="${ENTITY}"

# python -m exp.run \
#     --dataset=questions \
#     --d=4 \
#     --layers=4 \
#     --gnn_layers=1 \
#     --gnn_hidden=32 \
#     --pe_size=0 \
#     --hidden_channels=32 \
#     --epochs=2000 \
#     --early_stopping=200 \
#     --left_weights=True \
#     --right_weights=False \
#     --lr=0.002 \
#     --weight_decay=1e-7 \
#     --input_dropout=0.2 \
#     --dropout=0.2 \
#     --use_act=True \
#     --use_bias=True \
#     --folds=10 \
#     --model=CoopSheaf \
#     --normalised=True \
#     --sparse_learner=True \
#     --stop_strategy='acc' \
#     --entity="${ENTITY}"