#!/bin/bash
# shellcheck disable=SC1083,SC2016
python src/dataset_generate/generate_additional_target.py \
experiment=targets_generate/seq2seq_target_generator \
'dataset_folder="${paths.synth_data_dir}/regression/with_filler/different_target_count/sum/time_feature_min_trans_per_day=1_max_trans_per_day=3/num_applications=2000_num_transactions=4096_num_target_transactions_lower=90_num_target_transactions_upper=110_loc_target=1_scale_target=1_loc_first=1_scale_first=1/"'