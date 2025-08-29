#!/usr/bin/env python3
"""
Performance Evaluation Script for AI Models

This script evaluates the performance of AI models given various parameters including:
- Model configuration
- System configuration
- Total GPUs
- Parallelization settings
- Device planning
- Concurrency

It calculates key metrics:
- Tokens per second per GPU (tokens/s/gpu)
- Time To First Token (TTFT)
- Time Per Output Token (TPOT)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir / "src"))

from aiconfigurator.sdk import models, perf_database, config, common
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_parallel_config(parallel_str: str) -> Tuple[int, int, int, int, int]:
    """
    Parse parallel configuration string.
    
    Format: "tp<n>pp<n>dp<n>etp<n>ep<n>" or simplified formats
    Returns: (tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size)
    """
    # Default values
    tp_size = 1
    pp_size = 1
    dp_size = 1
    moe_tp_size = 1
    moe_ep_size = 1
    
    # Parse the string
    parts = parallel_str.lower().replace(' ', '')
    
    # Check for simplified format (e.g., "tp4", "tp2pp2")
    if 'tp' in parts:
        tp_match = parts.split('tp')[1].split('pp')[0].split('dp')[0].split('etp')[0].split('ep')[0]
        tp_size = int(tp_match) if tp_match else 1
    
    if 'pp' in parts:
        pp_match = parts.split('pp')[1].split('dp')[0].split('etp')[0].split('ep')[0]
        pp_size = int(pp_match) if pp_match else 1
    
    if 'dp' in parts:
        dp_match = parts.split('dp')[1].split('etp')[0].split('ep')[0]
        dp_size = int(dp_match) if dp_match else 1
    
    if 'etp' in parts:
        etp_match = parts.split('etp')[1].split('ep')[0]
        moe_tp_size = int(etp_match) if etp_match else 1
    
    if 'ep' in parts:
        ep_match = parts.split('ep')[1]
        moe_ep_size = int(ep_match) if ep_match else 1
    
    return tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size


def evaluate_performance(
    model_name: str,
    system: str,
    total_gpus: int,
    parallel: str,
    batch_size: int,
    isl: int,
    osl: int,
    concurrency: Optional[int] = None,
    backend_name: str = "trtllm",
    gemm_quant: str = "float16",
    kvcache_quant: str = "float16",
    fmha_quant: str = "float16",
    moe_quant: str = "float16",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model performance with given configuration.
    
    Returns dictionary with metrics:
    - tokens_per_second_per_gpu
    - ttft_ms (Time To First Token in milliseconds)
    - tpot_ms (Time Per Output Token in milliseconds)
    - total_tokens_per_second
    - sequences_per_second
    - latency_ms
    """
    logger = logging.getLogger(__name__)
    
    # Parse parallel configuration
    tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size = parse_parallel_config(parallel)
    
    # Validate GPU count
    required_gpus = tp_size * pp_size * dp_size
    if required_gpus != total_gpus:
        raise ValueError(f"Parallel config requires {required_gpus} GPUs but {total_gpus} specified")
    
    # Create model configuration
    model_config = config.ModelConfig(
        tp_size=tp_size,
        pp_size=pp_size,
        moe_tp_size=moe_tp_size if moe_tp_size > 1 else tp_size,
        moe_ep_size=moe_ep_size if moe_ep_size > 1 else 1,
        attention_dp_size=dp_size,
        gemm_quant_mode=getattr(common.GEMMQuantMode, gemm_quant),
        kvcache_quant_mode=getattr(common.KVCacheQuantMode, kvcache_quant),
        fmha_quant_mode=getattr(common.FMHAQuantMode, fmha_quant),
        moe_quant_mode=getattr(common.MoEQuantMode, moe_quant)
    )
    
    # Create runtime configuration
    runtime_config = config.RuntimeConfig(
        batch_size=batch_size,
        isl=isl,
        osl=osl,
        beam_width=1
    )
    
    # Get model
    model_obj = models.get_model(model_name, model_config)
    
    # Get database and backend with version
    # Default version based on backend
    default_versions = {
        'trtllm': '0.20.0',
        'vllm': 'v1',
        'sglang': 'v1'
    }
    
    version = default_versions.get(backend_name, '0.20.0')
    logger.debug(f"Using version: {version} for backend: {backend_name}")
    
    # Get database using the get_database function which handles paths correctly
    database = perf_database.get_database(system=system, backend=backend_name, version=version)
    if database is None:
        raise ValueError(f"Could not load database for system={system}, backend={backend_name}, version={version}")
    
    backend = get_backend(backend_name)
    
    # Create inference session
    session = InferenceSession(model=model_obj, database=database, backend=backend)
    
    # Run evaluation
    logger.info(f"Running evaluation for {model_name} on {system} with {total_gpus} GPUs")
    logger.info(f"Parallel config: TP={tp_size}, PP={pp_size}, DP={dp_size}, MoE-TP={moe_tp_size}, MoE-EP={moe_ep_size}")
    logger.info(f"Batch size: {batch_size}, ISL: {isl}, OSL: {osl}")
    
    # Run static inference to get performance metrics
    summary = session.run_static(runtime_config, mode='static')
    
    # Extract metrics from summary
    summary_df = summary.get_summary_df()
    
    # Get latency breakdowns
    context_latency_dict = summary.get_context_latency_dict()
    generation_latency_dict = summary.get_generation_latency_dict()
    
    # Calculate total latencies
    context_latency = sum(context_latency_dict.values())
    generation_latency = sum(generation_latency_dict.values())
    
    # Extract key metrics
    metrics = {
        'model': model_name,
        'system': system,
        'total_gpus': total_gpus,
        'parallel_config': parallel,
        'batch_size': batch_size,
        'isl': isl,
        'osl': osl,
        'ttft_ms': float(summary_df['ttft'].iloc[0]),
        'tpot_ms': float(summary_df['tpot'].iloc[0]),
        'tokens_per_second': float(summary_df['tokens/s'].iloc[0]),
        'tokens_per_second_per_gpu': float(summary_df['tokens/s/gpu'].iloc[0]),
        'sequences_per_second': float(summary_df['seq/s'].iloc[0]),
        'sequences_per_second_per_gpu': float(summary_df['seq/s/gpu'].iloc[0]),
        'total_latency_ms': float(summary_df['latency'].iloc[0]),
        'context_latency_ms': float(summary_df['context_latency'].iloc[0]),
        'generation_latency_ms': float(summary_df['generation_latency'].iloc[0]),
        'memory_gb': float(summary_df['memory'].iloc[0]),
        'is_oom': summary.check_oom()
    }
    
    # Add concurrency metrics if specified
    if concurrency:
        effective_batch = min(batch_size, concurrency)
        metrics['concurrency'] = concurrency
        metrics['effective_batch_size'] = effective_batch
        # Recalculate with concurrency consideration
        metrics['effective_tokens_per_second'] = metrics['tokens_per_second'] * (effective_batch / batch_size)
    
    # Add latency breakdown if verbose
    if verbose:
        metrics['context_breakdown'] = dict(context_latency_dict)
        metrics['generation_breakdown'] = dict(generation_latency_dict)
    
    return metrics


def print_results(metrics: Dict[str, Any], verbose: bool = False):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION RESULTS")
    print("="*60)
    
    # Basic configuration
    print(f"\nConfiguration:")
    print(f"  Model: {metrics['model']}")
    print(f"  System: {metrics['system']}")
    print(f"  Total GPUs: {metrics['total_gpus']}")
    print(f"  Parallel Config: {metrics['parallel_config']}")
    print(f"  Batch Size: {metrics['batch_size']}")
    print(f"  Input Sequence Length: {metrics['isl']}")
    print(f"  Output Sequence Length: {metrics['osl']}")
    
    if 'concurrency' in metrics:
        print(f"  Concurrency: {metrics['concurrency']}")
        print(f"  Effective Batch Size: {metrics['effective_batch_size']}")
    
    # Key metrics
    print(f"\nKey Metrics:")
    print(f"  Tokens/s/GPU: {metrics['tokens_per_second_per_gpu']:.2f}")
    print(f"  TTFT (ms): {metrics['ttft_ms']:.2f}")
    print(f"  TPOT (ms): {metrics['tpot_ms']:.2f}")
    
    # Throughput metrics
    print(f"\nThroughput:")
    print(f"  Total Tokens/s: {metrics['tokens_per_second']:.2f}")
    print(f"  Sequences/s: {metrics['sequences_per_second']:.2f}")
    print(f"  Sequences/s/GPU: {metrics['sequences_per_second_per_gpu']:.2f}")
    
    if 'effective_tokens_per_second' in metrics:
        print(f"  Effective Tokens/s (with concurrency): {metrics['effective_tokens_per_second']:.2f}")
    
    # Latency metrics
    print(f"\nLatency:")
    print(f"  Total Latency: {metrics['total_latency_ms']:.2f} ms")
    print(f"  Context Latency: {metrics['context_latency_ms']:.2f} ms")
    print(f"  Generation Latency: {metrics['generation_latency_ms']:.2f} ms")
    
    # Memory usage
    print(f"\nMemory:")
    print(f"  GPU Memory Usage: {metrics['memory_gb']:.2f} GB")
    print(f"  OOM Status: {'Yes' if metrics['is_oom'] else 'No'}")
    
    # Verbose breakdown
    if verbose and 'context_breakdown' in metrics:
        print(f"\nContext Phase Breakdown:")
        for op, latency in metrics['context_breakdown'].items():
            print(f"    {op}: {latency:.3f} ms")
        
        print(f"\nGeneration Phase Breakdown:")
        for op, latency in metrics['generation_breakdown'].items():
            print(f"    {op}: {latency:.3f} ms")
    
    print("\n" + "="*60)


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate AI model performance with given configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_performance.py --model GPT3_175B --system h100_sxm --gpus 8 --parallel tp8

  # With custom batch size and sequence lengths
  python evaluate_performance.py --model LLAMA3_70B --system h200_sxm --gpus 4 \\
      --parallel tp2pp2 --batch-size 16 --isl 2048 --osl 512

  # With concurrency and quantization
  python evaluate_performance.py --model DEEPSEEK_V3 --system gb200_nvl72 --gpus 72 \\
      --parallel tp8pp4dp2etp4ep2 --concurrency 32 --gemm-quant fp8_ootb

  # Verbose output with breakdown
  python evaluate_performance.py --model MIXTRAL_8X7B --system h100_sxm --gpus 2 \\
      --parallel tp2 --verbose
        """
    )
    
    # Required arguments
    parser.add_argument('--model', required=True, help='Model name (e.g., GPT3_175B, LLAMA3_70B)')
    parser.add_argument('--system', required=True, help='System configuration (e.g., h100_sxm, h200_sxm, gb200_nvl72)')
    parser.add_argument('--gpus', type=int, required=True, help='Total number of GPUs')
    parser.add_argument('--parallel', required=True, help='Parallel configuration (e.g., tp4, tp2pp2, tp8pp4dp2etp4ep2)')
    
    # Optional arguments
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--isl', type=int, default=128, help='Input sequence length (default: 128)')
    parser.add_argument('--osl', type=int, default=128, help='Output sequence length (default: 128)')
    parser.add_argument('--concurrency', type=int, help='Concurrency level for request handling')
    
    # Backend and quantization options
    parser.add_argument('--backend', default='trtllm', help='Backend to use (default: trtllm)')
    parser.add_argument('--gemm-quant', default='float16', help='GEMM quantization mode (default: float16)')
    parser.add_argument('--kvcache-quant', default='float16', help='KV cache quantization mode (default: float16)')
    parser.add_argument('--fmha-quant', default='float16', help='FMHA quantization mode (default: float16)')
    parser.add_argument('--moe-quant', default='float16', help='MoE quantization mode (default: float16)')
    
    # Output options
    parser.add_argument('--verbose', action='store_true', help='Show detailed breakdown')
    parser.add_argument('--output', help='Output file for results (CSV format)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    try:
        # Run evaluation
        metrics = evaluate_performance(
            model_name=args.model,
            system=args.system,
            total_gpus=args.gpus,
            parallel=args.parallel,
            batch_size=args.batch_size,
            isl=args.isl,
            osl=args.osl,
            concurrency=args.concurrency,
            backend_name=args.backend,
            gemm_quant=args.gemm_quant,
            kvcache_quant=args.kvcache_quant,
            fmha_quant=args.fmha_quant,
            moe_quant=args.moe_quant,
            verbose=args.verbose
        )
        
        # Print results
        print_results(metrics, verbose=args.verbose)
        
        # Save to file if requested
        if args.output:
            df = pd.DataFrame([metrics])
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        
        # Return success
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())