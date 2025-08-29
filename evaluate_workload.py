#!/usr/bin/env python3
"""
Evaluate workload performance using the aiconfigurator SDK.
Supports both unified and disaggregated configurations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir / "src"))

from aiconfigurator.sdk import models, perf_database, config, common
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession, DisaggInferenceSession


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_parallel_string(parallel: str) -> Tuple[int, int, int, int, int]:
    """Parse parallel configuration string like 'tp4pp1dp1etp4ep1'."""
    import re
    
    # Default values
    tp, pp, dp, etp, ep = 1, 1, 1, 1, 1
    
    # Parse each component
    if match := re.search(r'tp(\d+)', parallel):
        tp = int(match.group(1))
    if match := re.search(r'pp(\d+)', parallel):
        pp = int(match.group(1))
    if match := re.search(r'dp(\d+)', parallel):
        dp = int(match.group(1))
    if match := re.search(r'etp(\d+)', parallel):
        etp = int(match.group(1))
    if match := re.search(r'ep(\d+)', parallel):
        ep = int(match.group(1))
    
    return tp, pp, dp, etp, ep


def evaluate_unified(
    model_name: str,
    system: str,
    parallel: str,
    batch_size: int,
    isl: int,
    osl: int,
    backend_name: str = "trtllm",
    version: Optional[str] = None,
    quant_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Evaluate unified (non-disaggregated) configuration."""
    
    logger = logging.getLogger(__name__)
    
    # Parse parallel configuration
    tp, pp, dp, etp, ep = parse_parallel_string(parallel)
    
    # Create model configuration
    model_config = config.ModelConfig(
        tp_size=tp,
        pp_size=pp,
        moe_tp_size=etp if etp > 1 else tp,
        moe_ep_size=ep,
        attention_dp_size=dp,
        gemm_quant_mode=common.GEMMQuantMode[quant_config.get('gemm', 'float16')] if quant_config else common.GEMMQuantMode.float16,
        kvcache_quant_mode=common.KVCacheQuantMode[quant_config.get('kvcache', 'float16')] if quant_config else common.KVCacheQuantMode.float16,
        fmha_quant_mode=common.FMHAQuantMode[quant_config.get('fmha', 'float16')] if quant_config else common.FMHAQuantMode.float16,
        moe_quant_mode=common.MoEQuantMode[quant_config.get('moe', 'float16')] if quant_config else common.MoEQuantMode.float16
    )
    
    # Runtime configuration
    runtime_config = config.RuntimeConfig(
        batch_size=batch_size,
        isl=isl,
        osl=osl,
        beam_width=1
    )
    
    # Get database
    if version is None:
        version = {'trtllm': '0.20.0', 'vllm': 'v1', 'sglang': 'v1'}.get(backend_name, '0.20.0')
    
    database = perf_database.get_database(system=system, backend=backend_name, version=version)
    if database is None:
        # Force SOL mode if no database
        logger.warning(f"No performance database for {system}/{backend_name}/{version}, using SOL mode")
        database = perf_database.PerfDatabase(system=system, backend=backend_name, version=version)
        database._default_sol_mode = common.SOLMode.SOL
    
    backend = get_backend(backend_name)
    
    # Create model and session
    model = models.get_model(model_name, model_config)
    session = InferenceSession(model=model, database=database, backend=backend)
    
    # Run evaluation
    summary = session.run_static(runtime_config, mode='static')
    df = summary.get_summary_df()
    
    return {
        'type': 'unified',
        'summary_df': df,
        'context_latency': summary.get_context_latency_dict(),
        'generation_latency': summary.get_generation_latency_dict(),
        'is_oom': summary.check_oom()
    }


def evaluate_disaggregated(
    model_name: str,
    system: str,
    prefill_config: Dict,
    decode_config: Dict,
    isl: int,
    osl: int,
    backend_name: str = "trtllm",
    version: Optional[str] = None,
    quant_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Evaluate disaggregated configuration.
    
    prefill_config: {
        'workers': 12,
        'parallel': 'tp4pp1dp1etp4ep1',
        'batch_size': 1
    }
    decode_config: {
        'workers': 1,
        'parallel': 'tp1pp1dp16etp8ep2',
        'batch_size': 344
    }
    """
    
    logger = logging.getLogger(__name__)
    
    # Parse parallel configurations
    p_tp, p_pp, p_dp, p_etp, p_ep = parse_parallel_string(prefill_config['parallel'])
    d_tp, d_pp, d_dp, d_etp, d_ep = parse_parallel_string(decode_config['parallel'])
    
    # Create model configurations
    prefill_model_config = config.ModelConfig(
        tp_size=p_tp,
        pp_size=p_pp,
        moe_tp_size=p_etp if p_etp > 1 else p_tp,
        moe_ep_size=p_ep,
        attention_dp_size=p_dp,
        gemm_quant_mode=common.GEMMQuantMode[quant_config.get('gemm', 'float16')] if quant_config else common.GEMMQuantMode.float16,
        kvcache_quant_mode=common.KVCacheQuantMode[quant_config.get('kvcache', 'float16')] if quant_config else common.KVCacheQuantMode.float16,
        fmha_quant_mode=common.FMHAQuantMode[quant_config.get('fmha', 'float16')] if quant_config else common.FMHAQuantMode.float16,
        moe_quant_mode=common.MoEQuantMode[quant_config.get('moe', 'float16')] if quant_config else common.MoEQuantMode.float16
    )
    
    decode_model_config = config.ModelConfig(
        tp_size=d_tp,
        pp_size=d_pp,
        moe_tp_size=d_etp if d_etp > 1 else d_tp,
        moe_ep_size=d_ep,
        attention_dp_size=d_dp,
        gemm_quant_mode=common.GEMMQuantMode[quant_config.get('gemm', 'float16')] if quant_config else common.GEMMQuantMode.float16,
        kvcache_quant_mode=common.KVCacheQuantMode[quant_config.get('kvcache', 'float16')] if quant_config else common.KVCacheQuantMode.float16,
        fmha_quant_mode=common.FMHAQuantMode[quant_config.get('fmha', 'float16')] if quant_config else common.FMHAQuantMode.float16,
        moe_quant_mode=common.MoEQuantMode[quant_config.get('moe', 'float16')] if quant_config else common.MoEQuantMode.float16
    )
    
    # Get databases
    if version is None:
        version = {'trtllm': '0.20.0', 'vllm': 'v1', 'sglang': 'v1'}.get(backend_name, '0.20.0')
    
    database = perf_database.get_database(system=system, backend=backend_name, version=version)
    if database is None:
        # Force SOL mode if no database
        logger.warning(f"No performance database for {system}/{backend_name}/{version}, using SOL mode")
        database = perf_database.PerfDatabase(system=system, backend=backend_name, version=version)
        database._default_sol_mode = common.SOLMode.SOL
    
    backend = get_backend(backend_name)
    
    # Create disaggregated session
    disagg_session = DisaggInferenceSession(
        prefill_database=database,
        prefill_backend=backend,
        decode_database=database,
        decode_backend=backend
    )
    
    # Runtime configuration
    runtime_config = config.RuntimeConfig(
        isl=isl,
        osl=osl,
        beam_width=1
    )
    
    # Evaluate each component separately first
    logger.info("Evaluating prefill component...")
    prefill_model = models.get_model(model_name, prefill_model_config)
    prefill_session = InferenceSession(model=prefill_model, database=database, backend=backend)
    prefill_runtime = config.RuntimeConfig(batch_size=prefill_config['batch_size'], isl=isl, osl=osl, beam_width=1)
    prefill_summary = prefill_session.run_static(prefill_runtime, mode='static_ctx')
    
    logger.info("Evaluating decode component...")
    decode_model = models.get_model(model_name, decode_model_config)
    decode_session = InferenceSession(model=decode_model, database=database, backend=backend)
    decode_runtime = config.RuntimeConfig(batch_size=decode_config['batch_size'], isl=isl, osl=osl, beam_width=1)
    decode_summary = decode_session.run_static(decode_runtime, mode='static_gen')
    
    # Calculate combined metrics manually to avoid the Series issue
    prefill_df = prefill_summary.get_summary_df()
    decode_df = decode_summary.get_summary_df()
    
    # Extract scalar values properly
    p_seq_s = float(prefill_df['seq/s'].iloc[0])
    d_seq_s = float(decode_df['seq/s'].iloc[0])
    p_workers = prefill_config['workers']
    d_workers = decode_config['workers']
    
    # Calculate system throughput (limited by slower component)
    prefill_throughput = p_seq_s * p_workers
    decode_throughput = d_seq_s * d_workers
    system_seq_s = min(prefill_throughput, decode_throughput)
    
    # Calculate GPU counts
    p_gpus = p_tp * p_pp * p_dp
    d_gpus = d_tp * d_pp * d_dp
    total_gpus = p_gpus * p_workers + d_gpus * d_workers
    
    # Calculate metrics
    tokens_s = system_seq_s * osl
    tokens_s_gpu = tokens_s / total_gpus
    ttft = float(prefill_df['ttft'].iloc[0])
    tpot = float(decode_df['tpot'].iloc[0])
    tokens_s_user = 1000.0 / tpot if tpot > 0 else 0
    concurrency = float(decode_df['concurrency'].iloc[0]) * d_workers
    
    # Build combined summary DataFrame
    combined_data = {
        'model': model_name,
        'isl': isl,
        'osl': osl,
        'concurrency': concurrency,
        'request_rate': system_seq_s,
        'p_bs': prefill_config['batch_size'],
        'p_workers': p_workers,
        'd_bs': decode_config['batch_size'],
        'd_workers': d_workers,
        'ttft': ttft,
        'tpot': tpot,
        'seq/s': system_seq_s,
        'seq/s/gpu': system_seq_s / total_gpus,
        'tokens/s': tokens_s,
        'tokens/s/gpu': tokens_s_gpu,
        'tokens/s/user': tokens_s_user,
        'num_total_gpus': total_gpus,
        'p_seq/s/worker': p_seq_s,
        'd_seq/s/worker': d_seq_s,
        'bottleneck': 'prefill' if prefill_throughput < decode_throughput else 'decode',
        'system': system,
        'backend': backend_name,
        'version': version
    }
    
    combined_df = pd.DataFrame([combined_data])
    
    return {
        'type': 'disaggregated',
        'summary_df': combined_df,
        'prefill_df': prefill_df,
        'decode_df': decode_df,
        'prefill_throughput': prefill_throughput,
        'decode_throughput': decode_throughput,
        'bottleneck': combined_data['bottleneck']
    }


def print_results(result: Dict[str, Any], verbose: bool = False):
    """Print evaluation results."""
    
    print("\n" + "="*80)
    print("WORKLOAD EVALUATION RESULTS")
    print("="*80)
    
    df = result['summary_df']
    
    if result['type'] == 'unified':
        print("\nUnified Configuration:")
        print(f"  Model: {df['model'].iloc[0]}")
        print(f"  System: {df['system'].iloc[0]}")
        print(f"  Total GPUs: {df['num_total_gpus'].iloc[0]}")
        print(f"  Parallel: {df['parallel'].iloc[0]}")
        print(f"  Batch Size: {df['bs'].iloc[0]}")
    else:
        print("\nDisaggregated Configuration:")
        print(f"  Model: {df['model'].iloc[0]}")
        print(f"  System: {df['system'].iloc[0]}")
        print(f"  Total GPUs: {df['num_total_gpus'].iloc[0]}")
        print(f"  Prefill: {df['p_workers'].iloc[0]} workers, BS={df['p_bs'].iloc[0]}")
        print(f"  Decode: {df['d_workers'].iloc[0]} workers, BS={df['d_bs'].iloc[0]}")
        print(f"  Bottleneck: {df['bottleneck'].iloc[0]}")
    
    print(f"\nKey Performance Metrics:")
    print(f"  Tokens/s/GPU: {df['tokens/s/gpu'].iloc[0]:.2f}")
    print(f"  Tokens/s/user: {df['tokens/s/user'].iloc[0]:.2f}")
    print(f"  Total Tokens/s: {df['tokens/s'].iloc[0]:.2f}")
    print(f"  TTFT (ms): {df['ttft'].iloc[0]:.2f}")
    print(f"  TPOT (ms): {df['tpot'].iloc[0]:.2f}")
    print(f"  Sequences/s: {df['seq/s'].iloc[0]:.2f}")
    print(f"  Concurrency: {df['concurrency'].iloc[0]:.0f}")
    
    if verbose:
        print("\nFull Summary DataFrame:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.to_string(index=False))
    
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate workload performance using aiconfigurator SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unified configuration
  python evaluate_workload.py --model LLAMA3.1_70B --system h100_sxm \\
      --parallel tp4pp2 --batch-size 32 --isl 2048 --osl 512

  # Disaggregated configuration (the case from your table)
  python evaluate_workload.py --model DEEPSEEK_V3 --system gb200_nvl72 \\
      --disagg \\
      --prefill-workers 12 --prefill-parallel tp4pp1dp1etp4ep1 --prefill-bs 1 \\
      --decode-workers 1 --decode-parallel tp1pp1dp16etp8ep2 --decode-bs 344 \\
      --isl 128 --osl 128 \\
      --quant gemm=fp8_ootb,kvcache=fp8,fmha=fp8,moe=w4afp8
        """
    )
    
    # Model and system
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--system', required=True, help='System configuration')
    parser.add_argument('--backend', default='trtllm', help='Backend (default: trtllm)')
    parser.add_argument('--version', help='Backend version')
    
    # Sequence lengths
    parser.add_argument('--isl', type=int, default=128, help='Input sequence length')
    parser.add_argument('--osl', type=int, default=128, help='Output sequence length')
    
    # Unified configuration
    parser.add_argument('--parallel', help='Parallel configuration for unified mode')
    parser.add_argument('--batch-size', type=int, help='Batch size for unified mode')
    
    # Disaggregated configuration
    parser.add_argument('--disagg', action='store_true', help='Use disaggregated configuration')
    parser.add_argument('--prefill-workers', type=int, help='Number of prefill workers')
    parser.add_argument('--prefill-parallel', help='Prefill parallel configuration')
    parser.add_argument('--prefill-bs', type=int, help='Prefill batch size')
    parser.add_argument('--decode-workers', type=int, help='Number of decode workers')
    parser.add_argument('--decode-parallel', help='Decode parallel configuration')
    parser.add_argument('--decode-bs', type=int, help='Decode batch size')
    
    # Quantization
    parser.add_argument('--quant', help='Quantization config (e.g., gemm=fp8_ootb,kvcache=fp8)')
    
    # Output
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Parse quantization config
    quant_config = {}
    if args.quant:
        for item in args.quant.split(','):
            key, value = item.split('=')
            quant_config[key] = value
    
    try:
        if args.disagg:
            # Disaggregated mode
            if not all([args.prefill_workers, args.prefill_parallel, args.prefill_bs,
                       args.decode_workers, args.decode_parallel, args.decode_bs]):
                parser.error("Disaggregated mode requires all prefill and decode parameters")
            
            prefill_config = {
                'workers': args.prefill_workers,
                'parallel': args.prefill_parallel,
                'batch_size': args.prefill_bs
            }
            decode_config = {
                'workers': args.decode_workers,
                'parallel': args.decode_parallel,
                'batch_size': args.decode_bs
            }
            
            result = evaluate_disaggregated(
                model_name=args.model,
                system=args.system,
                prefill_config=prefill_config,
                decode_config=decode_config,
                isl=args.isl,
                osl=args.osl,
                backend_name=args.backend,
                version=args.version,
                quant_config=quant_config
            )
        else:
            # Unified mode
            if not all([args.parallel, args.batch_size]):
                parser.error("Unified mode requires --parallel and --batch-size")
            
            result = evaluate_unified(
                model_name=args.model,
                system=args.system,
                parallel=args.parallel,
                batch_size=args.batch_size,
                isl=args.isl,
                osl=args.osl,
                backend_name=args.backend,
                version=args.version,
                quant_config=quant_config
            )
        
        # Print results
        print_results(result, verbose=args.verbose)
        
        # Save to file if requested
        if args.output:
            result['summary_df'].to_csv(args.output, index=False)
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