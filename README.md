# Skoltech High Performance Python Lab 2024
## Student: Zabolotnyi Artem

```Run base implementation
bash scripts/data_generate/data_generate.sh
bash scripts/data_generate/seq2seq_target_generate.sh
```

```Run JIT base single target profiler
python tests/test_dataset_single_target_profiler.py
```

```Run JIT base multiple target profiler
python tests/test_dataset_multi_target_profiler.py
```


```Run JIT base single target per transaction
python tests/test_dataset_multi_target_single_target.py
```

```Run JIT base single target per transaction multiprocessing
python tests/test_dataset_multi_target_single_target_mp.py
```

```Run JIT base 20 target per transaction
python tests/test_dataset_multi_target_20_target.py
```

```Run JIT base 20 target per transaction multiprocessing
python tests/test_dataset_multi_target_20_target_mp.py
```

```Run profiler vizualization
snakeviz jit_pytorch_multi_target_load.pstats

snakeviz jit_pytorch_multi_target_load.pstats

snakeviz jit_pytorch_singe_target_load.pstats


```

