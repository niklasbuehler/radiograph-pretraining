#!/bin/sh
python3 src/train.py -m hparams_search=ma_vit_mae_optuna experiment=ma_vit_mae_hparams
