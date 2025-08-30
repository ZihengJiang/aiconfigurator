#!/usr/bin/env python3
"""
Evaluate Aggregated IFB (In-Flight Batching) Performance

This script provides detailed analysis of how IFB achieves its performance,
including operator breakdown, memory usage, and step-by-step execution.
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from collections import defaultdict

# AIConfigurator imports
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.perf_database import get_database, PerfDatabase
from aiconfigurator.sdk import models, config, common
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_ifb_steps(model, database, runtime_config, ctx_tokens: int) -> Dict:
    """Analyze IFB step-by-step execution"""
    
    isl = runtime_config.isl
    osl = runtime_config.osl
    b = runtime_config.batch_size
    
    # Calculate IFB parameters
    balance_score = isl * b / ctx_tokens / osl
    steps_to_finish_ctx = np.ceil(isl * b / ctx_tokens)
    
    analysis = {
        'ctx_tokens': ctx_tokens,
        'balance_score': balance_score,
        'steps_to_finish_ctx': steps_to_finish_ctx,
    }
    
    # Determine IFB mode and steps
    if b > 1:
        if steps_to_finish_ctx >= osl:
            # All mix mode
            num_mix_steps = steps_to_finish_ctx
            num_mix_ctx_tokens = ctx_tokens
            num_mix_gen_tokens = max(1, b//(steps_to_finish_ctx/osl))
            num_genonly_steps = 0
            num_genonly_tokens = 0
            mode = "All Mix"
        else:
            # Mix + Gen-only mode
            num_mix_steps = steps_to_finish_ctx
            num_mix_ctx_tokens = ctx_tokens
            num_mix_gen_tokens = b - np.ceil(ctx_tokens/isl)
            num_genonly_steps = osl - num_mix_steps
            num_genonly_tokens = b
            mode = "Mix + Gen-only"
    else:
        # Special case for b=1
        num_mix_steps = 1
        num_mix_ctx_tokens = ctx_tokens
        num_mix_gen_tokens = 0
        num_genonly_steps = osl - 1
        num_genonly_tokens = 1
        mode = "Single Request"
    
    analysis.update({
        'mode': mode,
        'num_mix_steps': num_mix_steps,
        'num_mix_ctx_tokens': num_mix_ctx_tokens,
        'num_mix_gen_tokens': num_mix_gen_tokens,
        'num_genonly_steps': num_genonly_steps,
        'num_genonly_tokens': num_genonly_tokens,
    })
    
    return analysis


def get_mix_step_breakdown(backend, model, database, ctx_tokens: int, gen_tokens: int, isl: int, osl: int) -> Dict:
    """Get detailed operator breakdown for a mix step"""
    
    # First pass: non-attention ops with ctx+gen tokens
    num_tokens = ctx_tokens + gen_tokens
    summary = backend.run_static(model, database, 
                                 config.RuntimeConfig(batch_size=1, beam_width=1, isl=num_tokens, osl=1), 
                                 mode='static_ctx')
    ctx_latency_dict = summary.get_context_latency_dict()
    
    non_attention_latency = 0.0
    non_attention_ops = {}
    for op_name, latency in ctx_latency_dict.items():
        if 'attention' not in op_name.lower():
            non_attention_latency += latency
            non_attention_ops[op_name] = latency
    
    # Second pass: context attention (averaged over steps)
    num_tokens = isl
    summary = backend.run_static(model, database,
                                config.RuntimeConfig(batch_size=1, beam_width=1, isl=num_tokens, osl=1),
                                mode='static_ctx')
    ctx_latency_dict = summary.get_context_latency_dict()
    ctx_attention_latency = ctx_latency_dict.get('context_attention', 0) / np.ceil(isl/ctx_tokens)
    
    # Third pass: generation attention (if there are gen tokens)
    gen_attention_latency = 0.0
    if gen_tokens > 0:
        summary = backend.run_static(model, database,
                                    config.RuntimeConfig(batch_size=gen_tokens, beam_width=1, isl=isl+osl//2, osl=2),
                                    mode='static_gen')
        gen_latency_dict = summary.get_generation_latency_dict()
        gen_attention_latency = gen_latency_dict.get('generation_attention', 0)
    
    breakdown = {
        'non_attention_ops': non_attention_ops,
        'non_attention_total': non_attention_latency,
        'context_attention': ctx_attention_latency,
        'generation_attention': gen_attention_latency,
        'total': non_attention_latency + ctx_attention_latency + gen_attention_latency
    }
    
    return breakdown


def get_genonly_step_breakdown(backend, model, database, gen_tokens: int, isl: int, osl: int) -> Dict:
    """Get detailed operator breakdown for a generation-only step"""
    
    if gen_tokens <= 0:
        return {'ops': {}, 'total': 0.0}
    
    summary = backend.run_static(model, database,
                                config.RuntimeConfig(batch_size=gen_tokens, beam_width=1, isl=isl+osl//2, osl=2),
                                mode='static_gen')
    gen_latency_dict = summary.get_generation_latency_dict()
    
    total = sum(gen_latency_dict.values())
    
    return {
        'ops': gen_latency_dict,
        'total': total
    }


def print_ifb_analysis(analysis: Dict, mix_breakdown: Dict, genonly_breakdown: Dict, 
                       ifb_result: Dict, memory_info: Dict):
    """Print comprehensive IFB analysis"""
    
    print("\n" + "="*80)
    print("IFB (IN-FLIGHT BATCHING) ANALYSIS")
    print("="*80)
    
    # Configuration
    print("\nConfiguration:")
    print(f"  Batch Size: {analysis['batch_size']}")
    print(f"  Input Sequence Length: {analysis['isl']}")
    print(f"  Output Sequence Length: {analysis['osl']}")
    print(f"  Context Chunk Size (ctx_tokens): {analysis['ctx_tokens']}")
    print(f"  Parallelism: {analysis['parallel']}")
    
    # IFB Mode Analysis
    print(f"\nIFB Mode: {analysis['mode']}")
    print(f"  Balance Score: {analysis['balance_score']:.2f}")
    print(f"  Steps to Finish Context: {analysis['steps_to_finish_ctx']:.0f}")
    
    # Step Breakdown
    print("\nStep Execution Plan:")
    if analysis['num_mix_steps'] > 0:
        print(f"  Mix Steps: {analysis['num_mix_steps']:.0f} steps")
        print(f"    - Context tokens per step: {analysis['num_mix_ctx_tokens']}")
        print(f"    - Generation tokens per step: {analysis['num_mix_gen_tokens']:.0f}")
        print(f"    - Latency per step: {mix_breakdown['total']:.2f}ms")
    
    if analysis['num_genonly_steps'] > 0:
        print(f"  Generation-only Steps: {analysis['num_genonly_steps']:.0f} steps")
        print(f"    - Generation tokens per step: {analysis['num_genonly_tokens']:.0f}")
        print(f"    - Latency per step: {genonly_breakdown['total']:.2f}ms")
    
    # Performance Metrics
    print("\nPerformance Metrics:")
    print(f"  TTFT: {ifb_result['ttft']:.2f}ms")
    print(f"  TPOT: {ifb_result['tpot']:.2f}ms")
    print(f"  Tokens/s/GPU: {ifb_result['tokens_s_gpu']:.2f}")
    print(f"  Tokens/s/user: {ifb_result['tokens_s_user']:.2f}")
    print(f"  Total Tokens/s: {ifb_result['tokens_s']:.2f}")
    
    # Memory Usage
    print("\nMemory Consumption:")
    print(f"  Per-GPU Memory:")
    for key, value in memory_info.items():
        if key != 'oom_status':
            print(f"    {key:<20}: {value:>10.3f} GB")
    print(f"  OOM Status: {'Yes' if memory_info.get('oom_status', False) else 'No'}")
    
    # Operator Breakdown - Mix Step
    if analysis['num_mix_steps'] > 0:
        print("\nMix Step Operator Breakdown:")
        print(f"  {'Operation':<30} {'Latency(ms)':>12} {'%Time':>8}")
        print("  " + "-"*50)
        
        # Non-attention ops
        total_time = mix_breakdown['total']
        for op_name, latency in sorted(mix_breakdown['non_attention_ops'].items(), 
                                      key=lambda x: x[1], reverse=True):
            percentage = (latency / total_time * 100) if total_time > 0 else 0
            print(f"  {op_name:<30} {latency:>12.3f} {percentage:>7.1f}%")
        
        # Attention ops
        if mix_breakdown['context_attention'] > 0:
            percentage = (mix_breakdown['context_attention'] / total_time * 100)
            print(f"  {'context_attention':<30} {mix_breakdown['context_attention']:>12.3f} {percentage:>7.1f}%")
        
        if mix_breakdown['generation_attention'] > 0:
            percentage = (mix_breakdown['generation_attention'] / total_time * 100)
            print(f"  {'generation_attention':<30} {mix_breakdown['generation_attention']:>12.3f} {percentage:>7.1f}%")
        
        print("  " + "-"*50)
        print(f"  {'Total':<30} {total_time:>12.3f} {100.0:>7.1f}%")
    
    # Operator Breakdown - Gen-only Step
    if analysis['num_genonly_steps'] > 0 and genonly_breakdown['total'] > 0:
        print("\nGeneration-only Step Operator Breakdown:")
        print(f"  {'Operation':<30} {'Latency(ms)':>12} {'%Time':>8}")
        print("  " + "-"*50)
        
        total_time = genonly_breakdown['total']
        for op_name, latency in sorted(genonly_breakdown['ops'].items(), 
                                      key=lambda x: x[1], reverse=True):
            percentage = (latency / total_time * 100) if total_time > 0 else 0
            print(f"  {op_name:<30} {latency:>12.3f} {percentage:>7.1f}%")
        
        print("  " + "-"*50)
        print(f"  {'Total':<30} {total_time:>12.3f} {100.0:>7.1f}%")
    
    # TTFT Calculation Explanation
    print("\nTTFT Calculation:")
    print(f"  TTFT = mix_step_latency × ceil(isl / ctx_tokens)")
    print(f"       = {mix_breakdown['total']:.2f}ms × ceil({analysis['isl']} / {analysis['ctx_tokens']})")
    print(f"       = {mix_breakdown['total']:.2f}ms × {np.ceil(analysis['isl']/analysis['ctx_tokens']):.0f}")
    print(f"       = {ifb_result['ttft']:.2f}ms")
    
    # TPOT Calculation Explanation
    if analysis['num_genonly_steps'] > 0:
        print("\nTPOT Calculation:")
        mix_contribution = analysis['num_mix_steps'] * mix_breakdown['total']
        gen_contribution = analysis['num_genonly_steps'] * genonly_breakdown['total']
        total_steps = analysis['num_mix_steps'] + analysis['num_genonly_steps']
        print(f"  TPOT = (mix_steps × mix_latency + gen_steps × gen_latency) / total_steps")
        print(f"       = ({analysis['num_mix_steps']:.0f} × {mix_breakdown['total']:.2f} + "
              f"{analysis['num_genonly_steps']:.0f} × {genonly_breakdown['total']:.2f}) / {total_steps:.0f}")
        print(f"       = {ifb_result['tpot']:.2f}ms")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Aggregated IFB Performance')
    parser.add_argument('--model', type=str, default='DEEPSEEK_V3', help='Model name')
    parser.add_argument('--system', type=str, default='gb200_nvl72', help='System name')
    parser.add_argument('--parallel', type=str, default='tp1pp1dp8etp2ep4', 
                       help='Parallelism configuration')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--isl', type=int, default=10240, help='Input sequence length')
    parser.add_argument('--osl', type=int, default=1024, help='Output sequence length')
    parser.add_argument('--ctx-tokens', type=int, default=51200, 
                       help='Context tokens per IFB step')
    parser.add_argument('--quant', type=str, default='gemm=fp8_block,kvcache=fp8,fmha=fp8,moe=fp8_block',
                       help='Quantization config')
    parser.add_argument('--backend', type=str, default='trtllm', help='Backend')
    parser.add_argument('--version', type=str, default='0.20.0', help='Backend version')
    
    args = parser.parse_args()
    
    # Parse parallelism configuration
    parallel_parts = args.parallel.replace('tp', '').replace('pp', 'p').replace('dp', 'd').replace('etp', 'e').replace('ep', 'e').split('p')
    tp = int(parallel_parts[0])
    pp = int(parallel_parts[1].split('d')[0])
    dp = int(parallel_parts[1].split('d')[1].split('e')[0])
    moe_tp = int(parallel_parts[1].split('d')[1].split('e')[1])
    moe_ep = int(parallel_parts[1].split('d')[1].split('e')[2])
    
    # Parse quantization config
    quant_config = {}
    if args.quant:
        for item in args.quant.split(','):
            key, value = item.split('=')
            quant_config[key] = value
    
    # Create model configuration
    model_config = config.ModelConfig(
        tp_size=tp,
        pp_size=pp,
        attention_dp_size=dp,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        gemm_quant_mode=common.GEMMQuantMode[quant_config.get('gemm', 'float16')],
        kvcache_quant_mode=common.KVCacheQuantMode[quant_config.get('kvcache', 'float16')],
        fmha_quant_mode=common.FMHAQuantMode[quant_config.get('fmha', 'float16')],
        moe_quant_mode=common.MoEQuantMode[quant_config.get('moe', 'float16')] if quant_config else common.MoEQuantMode.float16
    )
    
    # Create runtime configuration
    runtime_config = config.RuntimeConfig(
        batch_size=args.batch_size,
        isl=args.isl,
        osl=args.osl,
        beam_width=1
    )
    
    # Get database
    database = get_database(system=args.system, backend=args.backend, version=args.version)
    if database is None:
        logger.warning(f"Using SOL mode for {args.system}/{args.backend}/{args.version}")
        import os
        systems_dir = os.path.join(os.path.dirname(__file__), 'src', 'aiconfigurator', 'systems')
        database = PerfDatabase(system=args.system, backend=args.backend, version=args.version, systems_dir=systems_dir)
        database._default_sol_mode = common.SOLMode.SOL
    
    # Create model and backend
    model = models.get_model(args.model, model_config)
    backend = TRTLLMBackend()
    
    # Analyze IFB steps
    analysis = analyze_ifb_steps(model, database, runtime_config, args.ctx_tokens)
    analysis.update({
        'batch_size': args.batch_size,
        'isl': args.isl,
        'osl': args.osl,
        'parallel': args.parallel
    })
    
    # Get operator breakdowns
    mix_breakdown = get_mix_step_breakdown(
        backend, model, database,
        analysis['num_mix_ctx_tokens'],
        analysis['num_mix_gen_tokens'],
        args.isl, args.osl
    )
    
    genonly_breakdown = get_genonly_step_breakdown(
        backend, model, database,
        analysis['num_genonly_tokens'],
        args.isl, args.osl
    )
    
    # Run IFB to get actual results
    summary = backend.run_ifb(model, database, runtime_config, ctx_tokens=args.ctx_tokens)
    df = summary.get_summary_df()
    
    ifb_result = {
        'ttft': df['ttft'].iloc[0],
        'tpot': df['tpot'].iloc[0],
        'tokens_s_gpu': df['tokens/s/gpu'].iloc[0],
        'tokens_s_user': df['tokens/s/user'].iloc[0],
        'tokens_s': df['tokens/s'].iloc[0],
    }
    
    # Get memory info
    memory_info = summary._memory if hasattr(summary, '_memory') else {}
    
    # Print analysis
    print_ifb_analysis(analysis, mix_breakdown, genonly_breakdown, ifb_result, memory_info)


if __name__ == "__main__":
    main()