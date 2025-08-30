# Evaluate single replica (8 GPUs) matching AIConfigurator's best config
# Note: AIConfigurator deploys 8 such replicas for 64 GPUs total
# Each replica: 8 GPUs with tp1pp1dp8etp2ep4, batch_size=128
python evaluate_workload.py \
   --model DEEPSEEK_V3 \
   --system gb200_nvl72 \
   --parallel tp1pp1dp8etp2ep4 \
   --batch-size 128 \
   --isl 10240 \
   --osl 1024 \
   --quant gemm=fp8_block,kvcache=fp8,fmha=fp8,moe=fp8_block \
   --verbose