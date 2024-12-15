#!/bin/bash
# shellcheck disable=SC1083,SC2016
python -m cProfile -o jit.pstats tests/test_dataset_target_generation.py