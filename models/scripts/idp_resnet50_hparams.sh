#!/bin/bash
#source ../../idpvenv/bin/activate
python3 src/train.py experiment=idp_resnet50 hparams_search=idp_resnet50_optuna.yaml # debug=overfit
