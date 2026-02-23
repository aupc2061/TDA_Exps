#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python stream_benchmark.py \
  --config configs \
  --datasets I \
  --backbone RN50 \
  --stream-lengths 500,1000,2000,5000 \
  --cache-sizes 1,2,3,5
