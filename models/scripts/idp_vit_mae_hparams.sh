#!/bin/bash
#source ../../idpvenv/bin/activate
python3 src/train.py experiment=idp_vit_mae hparams_search=idp_vit_mae_optuna.yaml # debug=overfit
