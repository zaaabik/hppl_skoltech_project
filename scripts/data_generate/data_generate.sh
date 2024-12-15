#!/bin/bash
python src/dataset_generate/generate.py -m \
experiment=transactions_generate/normal_amount_mean_1 \
hparams_search='amount_reducer|target_transaction_ranges|transaction_in_a_day'
# hydra/launcher=joblib hydra.launcher.n_jobs=16
