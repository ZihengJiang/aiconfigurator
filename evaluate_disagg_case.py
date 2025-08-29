#!/usr/bin/env python3
"""
Evaluate Disaggregated Configuration Case

This script evaluates the specific disaggregated configuration where:
- Prefill workers: 12 workers with 4 GPUs each (tp4pp1dp1etp4ep1)
- Decode workers: 1 worker with 16 GPUs (tp1pp1dp16etp8ep2)
- Total: 64 GPUs
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir / "src"))

from aiconfigurator.sdk import models, perf_database, config, common
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import DisaggInferenceSession


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def evaluate_disagg_config(
    model_name: str = "DEEPSEEK_V3",
    system: str = "gb200_nvl72",
    isl: int = 128,
    osl: int = 128,
    backend_name: str = "trtllm",
    version: str = "0.20.0"
) -> Dict[str, Any]:
    """
    Evaluate the specific disaggregated configuration.
    
    Configuration:
    - 12 prefill workers with 4 GPUs each (tp4pp1dp1etp4ep1), batch_size=1
    - 1 decode worker with 16 GPUs (tp1pp1dp16etp8ep2), batch_size=344
    - Total: 64 GPUs
    """
    logger = logging.getLogger(__name__)
    
    # Prefill configuration
    prefill_tp = 4
    prefill_pp = 1
    prefill_dp = 1
    prefill_moe_tp = 4
    prefill_moe_ep = 1
    prefill_batch_size = 1
    prefill_num_workers = 12
    
    # Decode configuration
    decode_tp = 1
    decode_pp = 1
    decode_dp = 16
    decode_moe_tp = 8
    decode_moe_ep = 2
    decode_batch_size = 344
    decode_num_workers = 1
    
    # Calculate total GPUs
    prefill_gpus_per_worker = prefill_tp * prefill_pp * prefill_dp
    decode_gpus_per_worker = decode_tp * decode_pp * decode_dp
    total_gpus = prefill_gpus_per_worker * prefill_num_workers + decode_gpus_per_worker * decode_num_workers
    
    logger.info(f"Evaluating disaggregated configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  System: {system}")
    logger.info(f"  Total GPUs: {total_gpus}")
    logger.info(f"  Prefill: {prefill_num_workers} workers x {prefill_gpus_per_worker} GPUs, BS={prefill_batch_size}")
    logger.info(f"  Decode: {decode_num_workers} workers x {decode_gpus_per_worker} GPUs, BS={decode_batch_size}")
    
    # Create model configurations
    prefill_model_config = config.ModelConfig(
        tp_size=prefill_tp,
        pp_size=prefill_pp,
        moe_tp_size=prefill_moe_tp,
        moe_ep_size=prefill_moe_ep,
        attention_dp_size=prefill_dp,
        gemm_quant_mode=common.GEMMQuantMode.fp8_ootb,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.w4afp8
    )
    
    decode_model_config = config.ModelConfig(
        tp_size=decode_tp,
        pp_size=decode_pp,
        moe_tp_size=decode_moe_tp,
        moe_ep_size=decode_moe_ep,
        attention_dp_size=decode_dp,
        gemm_quant_mode=common.GEMMQuantMode.fp8_ootb,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.w4afp8
    )
    
    # Runtime configuration
    runtime_config = config.RuntimeConfig(
        isl=isl,
        osl=osl,
        beam_width=1
    )
    
    # Get databases and backends
    prefill_database = perf_database.get_database(system=system, backend=backend_name, version=version)
    decode_database = perf_database.get_database(system=system, backend=backend_name, version=version)
    
    if prefill_database is None or decode_database is None:
        raise ValueError(f"Could not load database for system={system}, backend={backend_name}, version={version}")
    
    backend = get_backend(backend_name)
    
    # Create disaggregated inference session
    disagg_session = DisaggInferenceSession(
        prefill_database=prefill_database,
        prefill_backend=backend,
        decode_database=decode_database,
        decode_backend=backend
    )
    
    # Run disaggregated evaluation
    logger.info("Running disaggregated inference evaluation...")
    
    result = disagg_session.run_disagg(
        model_name=model_name,
        runtime_config=runtime_config,
        prefill_model_config=prefill_model_config,
        prefill_batch_size=prefill_batch_size,
        prefill_num_worker=prefill_num_workers,
        decode_model_config=decode_model_config,
        decode_batch_size=decode_batch_size,
        decode_num_worker=decode_num_workers
    )
    
    # Extract metrics from result
    summary_df = result.get_summary_df()
    
    metrics = {
        'model': model_name,
        'system': system,
        'total_gpus': total_gpus,
        'prefill_workers': prefill_num_workers,
        'prefill_gpus_per_worker': prefill_gpus_per_worker,
        'prefill_batch_size': prefill_batch_size,
        'prefill_parallel': f'tp{prefill_tp}pp{prefill_pp}dp{prefill_dp}etp{prefill_moe_tp}ep{prefill_moe_ep}',
        'decode_workers': decode_num_workers,
        'decode_gpus_per_worker': decode_gpus_per_worker,
        'decode_batch_size': decode_batch_size,
        'decode_parallel': f'tp{decode_tp}pp{decode_pp}dp{decode_dp}etp{decode_moe_tp}ep{decode_moe_ep}',
        'isl': isl,
        'osl': osl,
        'ttft_ms': float(summary_df['ttft'].iloc[0]),
        'tpot_ms': float(summary_df['tpot'].iloc[0]),
        'tokens_per_second': float(summary_df['tokens/s'].iloc[0]),
        'tokens_per_second_per_gpu': float(summary_df['tokens/s/gpu'].iloc[0]),
        'tokens_per_second_per_user': float(summary_df['tokens/s/user'].iloc[0]),
        'sequences_per_second': float(summary_df['seq/s'].iloc[0]),
        'sequences_per_second_per_gpu': float(summary_df['seq/s/gpu'].iloc[0]),
        'concurrency': float(summary_df['concurrency'].iloc[0]),
        'request_rate': float(summary_df['request_rate'].iloc[0]),
        'prefill_seq_per_second_per_worker': float(summary_df['p_seq/s/worker'].iloc[0]),
        'decode_seq_per_second_per_worker': float(summary_df['d_seq/s/worker'].iloc[0])
    }
    
    return metrics, summary_df


def print_disagg_results(metrics: Dict[str, Any], summary_df: pd.DataFrame):
    """Print disaggregated evaluation results."""
    print("\n" + "="*80)
    print("DISAGGREGATED CONFIGURATION EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nConfiguration Overview:")
    print(f"  Model: {metrics['model']}")
    print(f"  System: {metrics['system']}")
    print(f"  Total GPUs: {metrics['total_gpus']}")
    
    print(f"\nPrefill Configuration:")
    print(f"  Workers: {metrics['prefill_workers']}")
    print(f"  GPUs per worker: {metrics['prefill_gpus_per_worker']}")
    print(f"  Total prefill GPUs: {metrics['prefill_workers'] * metrics['prefill_gpus_per_worker']}")
    print(f"  Batch size: {metrics['prefill_batch_size']}")
    print(f"  Parallel config: {metrics['prefill_parallel']}")
    print(f"  Seq/s per worker: {metrics['prefill_seq_per_second_per_worker']:.2f}")
    
    print(f"\nDecode Configuration:")
    print(f"  Workers: {metrics['decode_workers']}")
    print(f"  GPUs per worker: {metrics['decode_gpus_per_worker']}")
    print(f"  Total decode GPUs: {metrics['decode_workers'] * metrics['decode_gpus_per_worker']}")
    print(f"  Batch size: {metrics['decode_batch_size']}")
    print(f"  Parallel config: {metrics['decode_parallel']}")
    print(f"  Seq/s per worker: {metrics['decode_seq_per_second_per_worker']:.2f}")
    
    print(f"\nKey Performance Metrics:")
    print(f"  Tokens/s/GPU: {metrics['tokens_per_second_per_gpu']:.2f}")
    print(f"  Tokens/s/user: {metrics['tokens_per_second_per_user']:.2f}")
    print(f"  Total Tokens/s: {metrics['tokens_per_second']:.2f}")
    print(f"  TTFT (ms): {metrics['ttft_ms']:.2f}")
    print(f"  TPOT (ms): {metrics['tpot_ms']:.2f}")
    
    print(f"\nThroughput:")
    print(f"  Sequences/s: {metrics['sequences_per_second']:.2f}")
    print(f"  Sequences/s/GPU: {metrics['sequences_per_second_per_gpu']:.2f}")
    print(f"  Concurrency: {metrics['concurrency']:.0f}")
    print(f"  Request Rate: {metrics['request_rate']:.2f}")
    
    print(f"\nBottleneck Analysis:")
    prefill_throughput = metrics['prefill_seq_per_second_per_worker'] * metrics['prefill_workers']
    decode_throughput = metrics['decode_seq_per_second_per_worker'] * metrics['decode_workers']
    
    if prefill_throughput < decode_throughput:
        bottleneck = "Prefill"
        utilization = (prefill_throughput / decode_throughput) * 100
    else:
        bottleneck = "Decode"
        utilization = (decode_throughput / prefill_throughput) * 100
    
    print(f"  Bottleneck: {bottleneck}")
    print(f"  Prefill throughput: {prefill_throughput:.2f} seq/s")
    print(f"  Decode throughput: {decode_throughput:.2f} seq/s")
    print(f"  System utilization: {utilization:.1f}%")
    
    print("\n" + "="*80)
    
    # Display full summary dataframe
    print("\nFull Summary DataFrame:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(summary_df.to_string(index=False))


def compare_with_expected(metrics: Dict[str, Any]):
    """Compare results with expected values from the table."""
    expected = {
        'tokens_per_second_per_gpu': 1728.45,
        'tokens_per_second_per_user': 20.08,
        'concurrency': 5504,
        'total_gpus': 64
    }
    
    print("\n" + "="*80)
    print("COMPARISON WITH EXPECTED VALUES")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Expected':>15} {'Actual':>15} {'Difference':>15}")
    print("-" * 75)
    
    for key, expected_val in expected.items():
        if key in metrics:
            actual_val = metrics[key]
            diff = actual_val - expected_val
            diff_pct = (diff / expected_val) * 100 if expected_val != 0 else 0
            
            print(f"{key:<30} {expected_val:>15.2f} {actual_val:>15.2f} {diff:>+14.2f} ({diff_pct:+.1f}%)")
    
    print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate specific disaggregated configuration case"
    )
    
    parser.add_argument('--model', default='DEEPSEEK_V3', help='Model name (default: DEEPSEEK_V3)')
    parser.add_argument('--system', default='gb200_nvl72', help='System (default: gb200_nvl72)')
    parser.add_argument('--isl', type=int, default=128, help='Input sequence length (default: 128)')
    parser.add_argument('--osl', type=int, default=128, help='Output sequence length (default: 128)')
    parser.add_argument('--backend', default='trtllm', help='Backend (default: trtllm)')
    parser.add_argument('--version', default='0.20.0', help='Backend version (default: 0.20.0)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    try:
        # Run evaluation
        metrics, summary_df = evaluate_disagg_config(
            model_name=args.model,
            system=args.system,
            isl=args.isl,
            osl=args.osl,
            backend_name=args.backend,
            version=args.version
        )
        
        # Print results
        print_disagg_results(metrics, summary_df)
        
        # Compare with expected values
        compare_with_expected(metrics)
        
        # Save to file if requested
        if args.output:
            summary_df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())