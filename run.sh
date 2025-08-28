# aiconfigurator cli --model DEEPSEEK_V3 --total_gpus 64 --system h200_sxm --ttft 500 --tpot 50 --isl 10240 --osl 1024
aiconfigurator cli --model DEEPSEEK_V3 --total_gpus 64 --system gb200_nvl8 --ttft 500 --tpot 50 --isl 10240 --osl 1024 2>&1 | tee deepseek_gb200_nvl8.log
aiconfigurator cli --model DEEPSEEK_V3 --total_gpus 64 --system gb200_nvl72 --ttft 500 --tpot 50 --isl 10240 --osl 1024 2>&1 | tee deepseek_gb200_nvl72.log
