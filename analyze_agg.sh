#!/bin/bash

# Analyze aggregated configuration with IFB for DEEPSEEK_V3 on GB200 NVL72
# Configuration from top-performing setup: 2581.92 tokens/s/gpu

python analyze_agg.py \
    --model DEEPSEEK_V3 \
    --system gb200_nvl72 \
    --parallel tp1pp1dp8etp2ep4 \
    --batch-size 128 \
    --isl 10240 \
    --osl 1024 \
    --quant gemm=fp8_block,kvcache=fp8,fmha=fp8,moe=fp8_block \
    --ctx-tokens 51200