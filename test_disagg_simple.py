#!/usr/bin/env python3
"""
Simple test to evaluate the disaggregated configuration case.
Since gb200_nvl72 doesn't have performance data, it will use SOL fallback.
"""

import sys
from pathlib import Path

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir / "src"))

from aiconfigurator.sdk import models, perf_database, config, common
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession

def main():
    """Test the disaggregated case with simpler approach."""
    
    # Configuration from the table
    model_name = "DEEPSEEK_V3"
    system = "gb200_nvl72"
    isl = 128
    osl = 128
    
    # Since gb200_nvl72 doesn't have data, let's test with h100_sxm first
    # to verify the calculation logic
    system = "h100_sxm"  # Switch to a system with data
    
    print(f"Testing disaggregated configuration simulation on {system}")
    print("=" * 60)
    
    # Prefill configuration: 12 workers, 4 GPUs each (tp4)
    prefill_config = config.ModelConfig(
        tp_size=4,
        pp_size=1,
        moe_tp_size=4,
        moe_ep_size=1,
        attention_dp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_ootb,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.w4afp8
    )
    
    # Decode configuration: 1 worker, 16 GPUs (dp16)
    decode_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        moe_tp_size=8,
        moe_ep_size=2,
        attention_dp_size=16,
        gemm_quant_mode=common.GEMMQuantMode.fp8_ootb,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.w4afp8
    )
    
    # Runtime configuration
    runtime_config = config.RuntimeConfig(
        batch_size=1,  # Will be overridden
        isl=isl,
        osl=osl,
        beam_width=1
    )
    
    # Get database and backend
    database = perf_database.get_database(system=system, backend="trtllm", version="0.20.0")
    if database is None:
        print(f"Could not load database for {system}")
        return 1
    
    backend = get_backend("trtllm")
    
    # Test prefill performance (12 workers, BS=1)
    print("\nPrefill Performance (per worker):")
    prefill_model = models.get_model(model_name, prefill_config)
    prefill_session = InferenceSession(model=prefill_model, database=database, backend=backend)
    prefill_runtime = config.RuntimeConfig(batch_size=1, isl=isl, osl=osl, beam_width=1)
    prefill_summary = prefill_session.run_static(prefill_runtime, mode='static_ctx')
    prefill_df = prefill_summary.get_summary_df()
    
    print(f"  Batch size: 1")
    print(f"  Parallel: tp4pp1dp1")
    print(f"  TTFT: {prefill_df['ttft'].iloc[0]:.2f} ms")
    print(f"  Seq/s per worker: {prefill_df['seq/s'].iloc[0]:.2f}")
    print(f"  Total prefill throughput (12 workers): {prefill_df['seq/s'].iloc[0] * 12:.2f} seq/s")
    
    # Test decode performance (1 worker, BS=344)
    print("\nDecode Performance (per worker):")
    decode_model = models.get_model(model_name, decode_config)
    decode_session = InferenceSession(model=decode_model, database=database, backend=backend)
    decode_runtime = config.RuntimeConfig(batch_size=344, isl=isl, osl=osl, beam_width=1)
    decode_summary = decode_session.run_static(decode_runtime, mode='static_gen')
    decode_df = decode_summary.get_summary_df()
    
    print(f"  Batch size: 344")
    print(f"  Parallel: tp1pp1dp16")
    print(f"  TPOT: {decode_df['tpot'].iloc[0]:.2f} ms")
    print(f"  Seq/s per worker: {decode_df['seq/s'].iloc[0]:.2f}")
    print(f"  Concurrency: {decode_df['concurrency'].iloc[0]:.0f}")
    
    # Calculate combined metrics
    print("\nCombined Disaggregated Metrics:")
    prefill_throughput = prefill_df['seq/s'].iloc[0] * 12
    decode_throughput = decode_df['seq/s'].iloc[0] * 1
    
    # The system is limited by the slower component
    system_throughput = min(prefill_throughput, decode_throughput)
    total_gpus = 4 * 12 + 16 * 1  # 48 + 16 = 64
    
    tokens_per_second = system_throughput * osl
    tokens_per_second_per_gpu = tokens_per_second / total_gpus
    tokens_per_second_per_user = 1000.0 / decode_df['tpot'].iloc[0] if decode_df['tpot'].iloc[0] > 0 else 0
    
    print(f"  Total GPUs: {total_gpus}")
    print(f"  System throughput: {system_throughput:.2f} seq/s")
    print(f"  Tokens/s: {tokens_per_second:.2f}")
    print(f"  Tokens/s/GPU: {tokens_per_second_per_gpu:.2f}")
    print(f"  Tokens/s/user: {tokens_per_second_per_user:.2f}")
    print(f"  TTFT: {prefill_df['ttft'].iloc[0]:.2f} ms")
    print(f"  TPOT: {decode_df['tpot'].iloc[0]:.2f} ms")
    
    # Identify bottleneck
    print("\nBottleneck Analysis:")
    if prefill_throughput < decode_throughput:
        print(f"  Bottleneck: Prefill")
        print(f"  Utilization: {(prefill_throughput/decode_throughput)*100:.1f}%")
    else:
        print(f"  Bottleneck: Decode")  
        print(f"  Utilization: {(decode_throughput/prefill_throughput)*100:.1f}%")
    
    print("\nExpected values from table:")
    print(f"  Tokens/s/GPU: 1728.45")
    print(f"  Tokens/s/user: 20.08")
    print(f"  Concurrency: 5504")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())