#!/bin/bash

# AIConfigurator's actual deployment model:
# - 64 GPUs total = 8 replicas × 8 GPUs/replica
# - Each replica processes DIFFERENT requests (not the same batch)
# - Batch size 128 is the GLOBAL concurrent requests across all replicas
# - Each replica handles 128/8 = 16 requests

echo "==================================================================================="
echo "CORRECTED EVALUATION: What AIConfigurator Actually Measures"
echo "==================================================================================="
echo
echo "AIConfigurator Configuration:"
echo "  - 64 GPUs = 8 replicas × 8 GPUs/replica"
echo "  - Global batch: 128 concurrent requests"
echo "  - Per-replica batch: 16 requests"
echo "  - Parallelism per replica: tp1pp1dp8etp2ep4"
echo
echo "Evaluating single replica with correct batch size..."
echo

# Evaluate what each replica actually processes
python evaluate_workload.py \
   --model DEEPSEEK_V3 \
   --system gb200_nvl72 \
   --parallel tp1pp1dp8etp2ep4 \
   --batch-size 16 \
   --isl 10240 \
   --osl 1024 \
   --quant gemm=fp8_block,kvcache=fp8,fmha=fp8,moe=fp8_block

echo
echo "==================================================================================="
echo "To match AIConfigurator's aggregate metrics:"
echo "  - Total Tokens/s = above × 8 replicas"
echo "  - Tokens/s/GPU remains the same (work is replicated)"
echo "  - TTFT = single replica TTFT (each handles different requests)"
echo "  - TPOT = single replica TPOT"
echo "==================================================================================="