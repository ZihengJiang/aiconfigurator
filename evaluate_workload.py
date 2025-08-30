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


def get_op_details(model: Any, database: Any, runtime_config: Any, phase: str) -> Dict[str, Dict]:
    """Get detailed operation information including shapes, FLOPs, memory, and SOL values."""
    details_dict = {}
    batch_size = runtime_config.batch_size
    beam_width = runtime_config.beam_width
    isl = runtime_config.isl
    osl = runtime_config.osl
    
    # Get system specs for calculations
    system_spec = database.system_spec
    
    # Get the operations for the specified phase
    ops = model.context_ops if phase == 'context' else model.generation_ops
    
    for op in ops:
        op_details = {
            'sol_time': 0,
            'sol_math': 0,
            'sol_mem': 0,
            'flops': 0,
            'memory_bytes': 0,
            'shape': '',
            'compute_bound': False
        }
        
        # Query parameters
        if phase == 'context':
            x = batch_size * isl if 'logits_gemm' not in op._name else batch_size
            s = isl
        else:
            # Generation phase (with mtp/speculative decoding correction if needed)
            x = batch_size * beam_width * (model._nextn + 1)
            s = isl + osl // 2  # Average sequence length during generation
        
        # Get operation type and calculate details
        op_name = op._name
        
        if 'gemm' in op_name.lower():
            # GEMM operation
            from aiconfigurator.sdk.operations import GEMM
            if isinstance(op, GEMM):
                m = x
                n = op._n
                k = op._k
                op_details['shape'] = f"({m},{k})x({k},{n})"
                # FLOPs for single execution, will be multiplied by scale_factor (num layers) later
                single_flops = 2 * m * n * k  # 2 for multiply-add
                op_details['flops'] = single_flops * op._scale_factor  # Total FLOPs across all layers
                op_details['memory_bytes'] = op._quant_mode.value.memory * (m * n + m * k + n * k) * op._scale_factor
                
                # Get SOL values directly from database, not through op.query
                # to avoid scale_factor multiplication issues
                sol_math = 2 * m * n * k / (system_spec['gpu']['float16_tc_flops'] * op._quant_mode.value.compute) * 1000
                sol_mem = op._quant_mode.value.memory * (m * n + m * k + n * k) / system_spec['gpu']['mem_bw'] * 1000
                sol_time = max(sol_math, sol_mem)
                
                op_details['sol_time'] = sol_time * op._scale_factor
                op_details['sol_math'] = sol_math * op._scale_factor
                op_details['sol_mem'] = sol_mem * op._scale_factor
                op_details['compute_bound'] = sol_math > sol_mem
                    
        elif 'attention' in op_name.lower():
            # Attention operation - check if it's MLA first
            from aiconfigurator.sdk.operations import GenerationMLA, ContextMLA
            
            if isinstance(op, GenerationMLA):
                # GenerationMLA uses 1088 latent dimensions (DeepSeek V3)
                op_details['shape'] = f"B{x}xS{s}xH128"
                
                # Query returns total time for all layers already
                original_mode = database.get_default_sol_mode()
                database.set_default_sol_mode(common.SOLMode.SOL)
                op_details['sol_time'] = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
                database.set_default_sol_mode(original_mode)
                
                # Calculate FLOPs and memory for display
                # From query_generation_mla: ops = 2 * b * n * 1088 * s (per layer)
                tp_size = model.config.tp_size if hasattr(model, 'config') and hasattr(model.config, 'tp_size') else 1
                n = 128 // tp_size
                # GenerationMLA.query() returns total time (already multiplied by scale_factor)
                # For consistency, we need total FLOPs and memory too
                op_details['flops'] = 2 * x * n * 1088 * s * op._scale_factor
                # Memory: b * (n * 1088 * 2 + (s-1)*1088 * kvcache_bytes) per layer
                kvcache_bytes = 1 if hasattr(model, 'config') and hasattr(model.config, 'kvcache_quant_mode') and 'fp8' in str(model.config.kvcache_quant_mode) else 2
                # Total memory for all layers
                memory_per_layer = x * (n * 1088 * 2 + (s-1) * 1088 * kvcache_bytes)
                op_details['memory_bytes'] = memory_per_layer * op._scale_factor
                
            elif isinstance(op, ContextMLA):
                # ContextMLA for prefill phase
                op_details['shape'] = f"B{batch_size}xS{s}xH128"
                original_mode = database.get_default_sol_mode()
                database.set_default_sol_mode(common.SOLMode.SOL)
                op_details['sol_time'] = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
                database.set_default_sol_mode(original_mode)
                    
                # From query_context_mla: ops = b * 128 / tp_size * 2 / 2 *(s * s * 192 + s * s * 128)
                tp_size = model.config.tp_size if hasattr(model, 'config') and hasattr(model.config, 'tp_size') else 1
                n = 128 // tp_size
                op_details['flops'] = batch_size * n * 2 / 2 * (s * s * 192 + s * s * 128) * op._scale_factor
                op_details['memory_bytes'] = batch_size * n * 2 * (2*s*192 + 2*s*128) * op._scale_factor
                
            else:
                # Standard attention operation
                # Get total number of heads and TP size
                num_heads_total = getattr(model, '_n_heads', getattr(model, 'n_heads', 128))  # Total heads
                tp_size = model.config.tp_size if hasattr(model, 'config') and hasattr(model.config, 'tp_size') else 1
                # The operation receives per-GPU heads (already divided by TP)
                num_heads_per_gpu = num_heads_total // tp_size
                # PerfDatabase uses hardcoded 128 for head_dim
                head_dim_hardcoded = 128
                
                if phase == 'context':
                    op_details['shape'] = f"B{batch_size}xS{s}xH{num_heads_total}"
                    # Context attention FLOPs - per GPU calculation
                    # Formula matches PerfDatabase: ops = 2 * b * s * s * n * 128 * 2 / 2
                    # where n is per-GPU heads
                    per_gpu_flops = 2 * batch_size * s * s * num_heads_per_gpu * head_dim_hardcoded * 2 / 2
                    single_flops = per_gpu_flops  # Per layer, per GPU
                    op_details['flops'] = single_flops * op._scale_factor  # Total across all layers
                    
                    # Memory for Flash Attention (FMHA) - per GPU
                    # Formula matches PerfDatabase: mem_bytes = 2 * b * (n*s*128 + 2*n_kv*s*128 + n*s*128)
                    # where n is per-GPU heads
                    fmha_quant = model.config.fmha_quant_mode if hasattr(model, 'config') else common.FMHAQuantMode.float16
                    # Assume standard attention (n_kv = n)
                    n_kv_per_gpu = num_heads_per_gpu
                    # Memory: 2 * b * (n*s*128 + 2*n_kv*s*128 + n*s*128) where n is per-GPU
                    mem_bytes_per_gpu = 2 * batch_size * (
                        num_heads_per_gpu * s * head_dim_hardcoded +
                        2 * n_kv_per_gpu * s * head_dim_hardcoded +
                        num_heads_per_gpu * s * head_dim_hardcoded
                    )
                    op_details['memory_bytes'] = mem_bytes_per_gpu * op._scale_factor
                else:
                    op_details['shape'] = f"B{x}xS{s}xH{num_heads_total}"
                    # Generation attention FLOPs - per GPU calculation
                    # Formula matches PerfDatabase: ops = 2 * b * n * 128 * s * 2
                    # where n is per-GPU heads
                    per_gpu_flops = 2 * x * num_heads_per_gpu * head_dim_hardcoded * s * 2
                    single_flops = per_gpu_flops  # Per layer, per GPU
                    op_details['flops'] = single_flops * op._scale_factor  # Total across all layers
                    
                    # Memory for generation - per GPU
                    # Formula matches PerfDatabase: b * (n*128*2 + 2*n_kv*(s-1)*128*kvcache_quant_mode.value.memory + n*128*2)
                    # where n and n_kv are per-GPU heads
                    kvcache_quant = model.config.kvcache_quant_mode if hasattr(model, 'config') else common.KVCacheQuantMode.float16
                    kv_bytes_per_elem = kvcache_quant.value.memory if hasattr(kvcache_quant.value, 'memory') else 2
                    
                    # Assume standard attention (n_kv = n) - both are per-GPU
                    n_kv_per_gpu = num_heads_per_gpu
                    
                    # Memory components per GPU:
                    # 1. Query input (FP16): x * n * 128 * 2 bytes (n is per-GPU)
                    query_memory = x * num_heads_per_gpu * head_dim_hardcoded * 2
                    # 2. KV cache read from previous positions: 2 * x * n_kv * (s-1) * 128 * kv_bytes
                    kv_cache_memory = 2 * x * n_kv_per_gpu * max(s-1, 0) * head_dim_hardcoded * kv_bytes_per_elem
                    # 3. Output (FP16): x * n * 128 * 2 bytes  
                    output_memory = x * num_heads_per_gpu * head_dim_hardcoded * 2
                    
                    mem_bytes_per_gpu = query_memory + kv_cache_memory + output_memory
                    op_details['memory_bytes'] = mem_bytes_per_gpu * op._scale_factor
                
                # Get SOL for attention
                original_mode = database.get_default_sol_mode()
                database.set_default_sol_mode(common.SOLMode.SOL)
                op_details['sol_time'] = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
                database.set_default_sol_mode(original_mode)
            
        elif 'moe' in op_name.lower():
            # MoE and MoE dispatch operations
            from aiconfigurator.sdk.operations import MoE, MoEDispatch
            if isinstance(op, MoE):
                op_details['shape'] = f"E{op._num_experts}xK{op._topk}x({x},{op._hidden_size},{op._inter_size})"
                # MoE FLOPs with SwiGLU: 3 matrices (gate, up, down), each with 2 ops (multiply-add)
                # Total FLOPs = 3 * 2 * topk * x * hidden * inter_size
                # This is the total work, but it's divided across moe_tp_size GPUs
                total_flops = 3 * 2 * op._topk * x * op._hidden_size * op._inter_size
                # Per-GPU FLOPs (work is divided by moe_tp_size)
                per_gpu_flops = total_flops / op._moe_tp_size if hasattr(op, '_moe_tp_size') and op._moe_tp_size > 0 else total_flops
                single_flops = per_gpu_flops  # This is per-layer, per-GPU
                op_details['flops'] = single_flops * op._scale_factor  # Total across all layers
                # Memory calculation matching the SOL calculation in PerfDatabase.query_moe
                # This includes input/output, intermediate activations, and weight loading
                # MoE operation scales x by attention_dp_size, so use the same scaling
                x_scaled = x * op._attention_dp_size
                total_tokens = x_scaled * op._topk
                moe_tp = op._moe_tp_size if hasattr(op, '_moe_tp_size') and op._moe_tp_size > 0 else 1
                moe_ep = op._moe_ep_size if hasattr(op, '_moe_ep_size') and op._moe_ep_size > 0 else 1
                num_experts = op._num_experts if hasattr(op, '_num_experts') else 256
                
                mem_bytes_per_layer = op._quant_mode.value.memory * (
                    total_tokens * op._hidden_size * 3  # input + output
                    + total_tokens * op._inter_size * 3 // moe_tp  # intermediate activations
                    + op._hidden_size * op._inter_size * 3 // moe_tp * min(num_experts // moe_ep, total_tokens)  # weights loading
                )
                op_details['memory_bytes'] = mem_bytes_per_layer * op._scale_factor  # Total across all layers
                
                # Get SOL
                original_mode = database.get_default_sol_mode()
                database.set_default_sol_mode(common.SOLMode.SOL)
                op_details['sol_time'] = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
                database.set_default_sol_mode(original_mode)
            elif isinstance(op, MoEDispatch):
                # MoE dispatch operation (includes communication)
                op_details['shape'] = f"Dispatch({x} tokens, E{op._num_experts})"
                # Dispatch involves routing and communication
                original_mode = database.get_default_sol_mode()
                database.set_default_sol_mode(common.SOLMode.SOL)
                op_details['sol_time'] = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
                database.set_default_sol_mode(original_mode)
            else:
                # Unknown MoE-related operation
                op_details['shape'] = f"({x},...)"
                original_mode = database.get_default_sol_mode()
                database.set_default_sol_mode(common.SOLMode.SOL)
                op_details['sol_time'] = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
                database.set_default_sol_mode(original_mode)
                
        elif 'norm' in op_name.lower():
            # Layer norm operation
            op_details['shape'] = f"({x},{model._hidden_size})"
            op_details['flops'] = x * model._hidden_size * 5  # mean, var, normalize, scale, bias
            op_details['memory_bytes'] = 2 * x * model._hidden_size * 2  # input + output
            op_details['sol_time'] = op_details['memory_bytes'] / system_spec['gpu']['mem_bw'] * 1000
            op_details['sol_mem'] = op_details['sol_time']
            
        elif 'mla' in op_name.lower() or 'bmm' in op_name.lower():
            # MLA (Multi-head Latent Attention) operations
            # These are DeepSeek-specific operations that use latent dimensions
            from aiconfigurator.sdk.operations import GenerationMLA, ContextMLA, MLABmm
            
            if isinstance(op, GenerationMLA):
                # GenerationMLA uses 1088 latent dimensions
                op_details['shape'] = f"B{x}xS{s}xH128"
                # Query returns total time for all layers already
                original_mode = database.get_default_sol_mode()
                database.set_default_sol_mode(common.SOLMode.SOL_FULL)
                sol_result = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
                database.set_default_sol_mode(original_mode)
                
                if isinstance(sol_result, tuple):
                    op_details['sol_time'] = sol_result[0]
                    op_details['sol_math'] = sol_result[1]
                    op_details['sol_mem'] = sol_result[2]
                else:
                    op_details['sol_time'] = sol_result
                
                # Calculate FLOPs and memory for display
                # From query_generation_mla: ops = 2 * b * n * 1088 * s (per layer)
                tp_size = model.config.tp_size if hasattr(model, 'config') and hasattr(model.config, 'tp_size') else 1
                n = 128 // tp_size
                # GenerationMLA.query() returns total time (already multiplied by scale_factor)
                # For consistency, we need total FLOPs and memory too
                op_details['flops'] = 2 * x * n * 1088 * s * op._scale_factor
                # Memory: b * (n * 1088 * 2 + (s-1)*1088 * kvcache_bytes) per layer
                kvcache_bytes = 1 if hasattr(model, 'config') and hasattr(model.config, 'kvcache_quant_mode') and 'fp8' in str(model.config.kvcache_quant_mode) else 2
                # Total memory for all layers
                memory_per_layer = x * (n * 1088 * 2 + (s-1) * 1088 * kvcache_bytes)
                op_details['memory_bytes'] = memory_per_layer * op._scale_factor
                print(f"DEBUG GenerationMLA: x={x}, n={n}, s={s}, scale_factor={op._scale_factor}")
                print(f"DEBUG memory_per_layer={memory_per_layer:,}, total={op_details['memory_bytes']:,}")
                
            elif isinstance(op, ContextMLA):
                # ContextMLA for prefill phase
                op_details['shape'] = f"B{batch_size}xS{s}xH128"
                original_mode = database.get_default_sol_mode()
                database.set_default_sol_mode(common.SOLMode.SOL_FULL)
                sol_result = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
                database.set_default_sol_mode(original_mode)
                
                if isinstance(sol_result, tuple):
                    op_details['sol_time'] = sol_result[0]
                    op_details['sol_math'] = sol_result[1]
                    op_details['sol_mem'] = sol_result[2]
                else:
                    op_details['sol_time'] = sol_result
                    
                # From query_context_mla: ops = b * 128 / tp_size * 2 / 2 *(s * s * 192 + s * s * 128)
                tp_size = model.config.tp_size if hasattr(model, 'config') and hasattr(model.config, 'tp_size') else 1
                n = 128 // tp_size
                op_details['flops'] = batch_size * n * 2 / 2 * (s * s * 192 + s * s * 128) * op._scale_factor
                op_details['memory_bytes'] = batch_size * n * 2 * (2*s*192 + 2*s*128) * op._scale_factor
                
            elif isinstance(op, MLABmm):
                # MLA BMM operations
                op_details['shape'] = f"({x},...)"
                original_mode = database.get_default_sol_mode()
                database.set_default_sol_mode(common.SOLMode.SOL)
                op_details['sol_time'] = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
                database.set_default_sol_mode(original_mode)
            else:
                # Other MLA-related ops
                op_details['shape'] = f"({x},...)"
                original_mode = database.get_default_sol_mode()
                database.set_default_sol_mode(common.SOLMode.SOL)
                op_details['sol_time'] = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
                database.set_default_sol_mode(original_mode)
                
        else:
            # Default/other operations
            op_details['shape'] = f"({x},...)"
            original_mode = database.get_default_sol_mode()
            database.set_default_sol_mode(common.SOLMode.SOL)
            op_details['sol_time'] = op.query(database, x=x, batch_size=batch_size, beam_width=beam_width, s=s)
            database.set_default_sol_mode(original_mode)
        
        # Calculate utilization rates (this will be done later when we have actual times)
        op_details['compute_util'] = 0
        op_details['memory_util'] = 0
        op_details['utilization'] = 0
        
        details_dict[op._name] = op_details
    
    return details_dict


def update_op_details_with_utilization(op_details: Dict[str, Dict], actual_times: Dict[str, float]) -> Dict[str, Dict]:
    """Update operation details with utilization calculations based on actual times."""
    for op_name, details in op_details.items():
        actual_time = actual_times.get(op_name, 0)
        if actual_time > 0 and 'sol_time' in details and details['sol_time'] > 0:
            details['utilization'] = (details['sol_time'] / actual_time) * 100
            
            if 'sol_math' in details and 'sol_mem' in details:
                details['compute_util'] = (details['sol_math'] / actual_time) * 100
                details['memory_util'] = (details['sol_mem'] / actual_time) * 100
            else:
                details['compute_util'] = details['utilization']
                details['memory_util'] = details['utilization']
    return op_details


def get_sol_latencies(model: Any, database: Any, runtime_config: Any, phase: str) -> Dict[str, float]:
    """Get SOL (Speed of Light) latencies for operations."""
    details = get_op_details(model, database, runtime_config, phase)
    # Filter out operations with 0 time (they might not be computed in SOL mode)
    return {op_name: details['sol_time'] for op_name, details in details.items() if details['sol_time'] > 0}


def get_operation_latencies(model: Any, database: Any, runtime_config: Any, phase: str, use_sol: bool = False) -> Dict[str, float]:
    """Get operation latencies - either from database or SOL calculations."""
    if use_sol:
        # Use SOL calculations directly
        return get_sol_latencies(model, database, runtime_config, phase)
    else:
        # Get from session/database
        session = InferenceSession(model=model, database=database, backend=get_backend('trtllm'))
        if phase == 'context':
            summary = session.run_static(runtime_config, mode='static_ctx')
            return summary.get_context_latency_dict()
        else:
            summary = session.run_static(runtime_config, mode='static_gen')
            return summary.get_generation_latency_dict()


def calculate_op_utilizations(actual_dict: Dict[str, float], sol_dict: Dict[str, float]) -> Dict[str, float]:
    """Calculate utilization percentages for each operation."""
    utilization_dict = {}
    
    for op_name, actual_time in actual_dict.items():
        if op_name in sol_dict and sol_dict[op_name] > 0:
            # Utilization = SOL time / Actual time * 100
            # Higher utilization means closer to theoretical peak
            utilization = (sol_dict[op_name] / actual_time * 100) if actual_time > 0 else 0
            utilization = min(utilization, 100)  # Cap at 100%
        else:
            utilization = 0
        
        utilization_dict[op_name] = utilization
    
    return utilization_dict


def add_communication_ops(breakdown: Dict[str, float], model: Any, database: Any, runtime: Any, phase: str) -> Dict[str, float]:
    """Add communication operations (AllReduce, NCCL) to the breakdown."""
    # Check if we already have communication ops
    has_comm = any('ar_' in k or 'allreduce' in k.lower() for k in breakdown.keys())
    
    if not has_comm and hasattr(model, 'config') and hasattr(model.config, 'tp_size') and model.config.tp_size > 1:
        # Estimate communication time based on TP size and data volume
        # For DeepSeek and other models without explicit AllReduce ops
        
        # Get model parameters
        hidden_size = model._hidden_size if hasattr(model, '_hidden_size') else 7168
        num_layers = model._num_layers if hasattr(model, '_num_layers') else 61
        tp_size = model.config.tp_size
        batch_size = runtime.batch_size if hasattr(runtime, 'batch_size') else 1
        seq_len = runtime.isl if phase == 'context' else 1
        
        # Estimate AllReduce data volume per layer (hidden_size * batch * seq_len * 2 bytes for FP16)
        data_volume_per_layer_mb = (hidden_size * batch_size * seq_len * 2) / (1024 * 1024)
        
        # Use NCCL bandwidth from system spec if available
        if hasattr(database, '_system_spec') and 'interconnect' in database._system_spec:
            # GB200 NVL72 has 1.8TB/s NVLink bandwidth per GPU, 900GB/s bisection
            nccl_bandwidth_gb_s = 900  # GB/s effective for AllReduce on NVL72
        else:
            nccl_bandwidth_gb_s = 100  # Default assumption
        
        # Calculate communication time per AllReduce
        # AllReduce algorithm: ring-allreduce uses 2*(tp_size-1)/tp_size of data volume
        allreduce_factor = 2 * (tp_size - 1) / tp_size
        comm_time_per_op_ms = (data_volume_per_layer_mb / 1024 * allreduce_factor / nccl_bandwidth_gb_s) * 1000
        
        # Add two AllReduce operations per layer (typical for transformer: after attention and FFN)
        # Total time = time_per_op * num_layers
        if phase == 'context':
            breakdown['context_ar_1'] = comm_time_per_op_ms * num_layers
            breakdown['context_ar_2'] = comm_time_per_op_ms * num_layers
        else:
            breakdown['generation_ar_1'] = comm_time_per_op_ms * num_layers  
            breakdown['generation_ar_2'] = comm_time_per_op_ms * num_layers
    
    return breakdown


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
    # For unified mode with DP, the effective batch size per GPU is batch_size / dp
    effective_batch_size = batch_size // dp if dp > 1 else batch_size
    runtime_config = config.RuntimeConfig(
        batch_size=effective_batch_size,
        isl=isl,
        osl=osl,
        beam_width=1
    )
    
    # Get database
    if version is None:
        version = {'trtllm': '0.20.0', 'vllm': 'v1', 'sglang': 'v1'}.get(backend_name, '0.20.0')
    
    database = perf_database.get_database(system=system, backend=backend_name, version=version)
    use_sol_mode = False
    if database is None or system == 'gb200_nvl72':
        # Force SOL mode if no database or for GB200 NVL72
        logger.warning(f"Using SOL mode for {system}/{backend_name}/{version}")
        import os
        systems_dir = os.path.join(os.path.dirname(__file__), 'src', 'aiconfigurator', 'systems')
        database = perf_database.PerfDatabase(system=system, backend=backend_name, version=version, systems_dir=systems_dir)
        database._default_sol_mode = common.SOLMode.SOL
        use_sol_mode = True
    
    backend = get_backend(backend_name)
    
    # Create model and session
    model = models.get_model(model_name, model_config)
    session = InferenceSession(model=model, database=database, backend=backend)
    
    # Run evaluation
    summary = session.run_static(runtime_config, mode='static')
    df = summary.get_summary_df()
    
    # Get memory information from summary
    memory_info = {}
    if hasattr(summary, '_memory') and summary._memory:
        memory_info = summary._memory.copy()
        memory_info['oom_status'] = summary.check_oom()
    elif 'memory' in df.columns:
        memory_info = {'total': float(df['memory'].iloc[0]), 'oom_status': summary.check_oom()}
    
    # If still empty, try to get from summary methods
    if not memory_info and hasattr(summary, 'get_memory_consumption'):
        mem_dict = summary.get_memory_consumption()
        if mem_dict:
            memory_info = mem_dict
            memory_info['oom_status'] = summary.check_oom()
    
    # Get operation latencies - use SOL if needed
    if use_sol_mode:
        context_dict = get_operation_latencies(model, database, runtime_config, 'context', use_sol=True)
        gen_dict = get_operation_latencies(model, database, runtime_config, 'generation', use_sol=True)
    else:
        context_dict = summary.get_context_latency_dict()
        gen_dict = summary.get_generation_latency_dict()
    
    context_total = sum(context_dict.values()) if context_dict else 0
    gen_total = sum(gen_dict.values()) if gen_dict else 0
    
    # Calculate SOL (Speed of Light) values for utilization
    context_sol_dict = get_sol_latencies(model, database, runtime_config, 'context')
    gen_sol_dict = get_sol_latencies(model, database, runtime_config, 'generation')
    
    # Calculate utilization for each operation
    context_utilization = calculate_op_utilizations(context_dict, context_sol_dict)
    gen_utilization = calculate_op_utilizations(gen_dict, gen_sol_dict)
    
    # Get detailed operation information
    context_op_details = get_op_details(model, database, runtime_config, 'context')
    gen_op_details = get_op_details(model, database, runtime_config, 'generation')
    
    # Update with utilization based on actual times
    context_op_details = update_op_details_with_utilization(context_op_details, context_dict)
    gen_op_details = update_op_details_with_utilization(gen_op_details, gen_dict)
    
    return {
        'type': 'unified',
        'summary_df': df,
        'context_latency': context_dict,
        'generation_latency': gen_dict,
        'context_total_time': context_total,
        'generation_total_time': gen_total,
        'context_sol': context_sol_dict,
        'generation_sol': gen_sol_dict,
        'context_utilization': context_utilization,
        'generation_utilization': gen_utilization,
        'context_op_details': context_op_details,
        'generation_op_details': gen_op_details,
        'memory_info': memory_info,
        'is_oom': summary.check_oom(),
        'model': model,
        'sol_mode': use_sol_mode,
        'system': system,
        'batch_size': batch_size,  # Original batch size
        'effective_batch_size': effective_batch_size,  # Per-GPU batch size
        'dp': dp,  # Data parallelism factor
        'parallel': parallel  # Full parallel config
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
    use_sol_mode = False
    if database is None or system == 'gb200_nvl72':
        # Force SOL mode if no database or for GB200 NVL72
        logger.warning(f"Using SOL mode for {system}/{backend_name}/{version}")
        import os
        systems_dir = os.path.join(os.path.dirname(__file__), 'src', 'aiconfigurator', 'systems')
        database = perf_database.PerfDatabase(system=system, backend=backend_name, version=version, systems_dir=systems_dir)
        database._default_sol_mode = common.SOLMode.SOL
        use_sol_mode = True
    
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
    
    # Get memory information
    memory_info = {
        'prefill': {},
        'decode': {},
        'prefill_workers': p_workers,
        'decode_workers': d_workers
    }
    
    if hasattr(prefill_summary, '_memory') and prefill_summary._memory:
        memory_info['prefill'] = prefill_summary._memory.copy()
    elif 'memory' in prefill_df.columns:
        memory_info['prefill'] = {'total': float(prefill_df['memory'].iloc[0])}
    
    if hasattr(decode_summary, '_memory') and decode_summary._memory:
        memory_info['decode'] = decode_summary._memory.copy()
    elif 'memory' in decode_df.columns:
        memory_info['decode'] = {'total': float(decode_df['memory'].iloc[0])}
    
    # Get operation breakdowns - use SOL if needed
    if use_sol_mode:
        prefill_breakdown = get_operation_latencies(prefill_model, database, prefill_runtime, 'context', use_sol=True)
        decode_breakdown = get_operation_latencies(decode_model, database, decode_runtime, 'generation', use_sol=True)
    else:
        prefill_breakdown = prefill_summary.get_context_latency_dict()
        decode_breakdown = decode_summary.get_generation_latency_dict()
    
    # Calculate SOL values for utilization
    prefill_sol = get_sol_latencies(prefill_model, database, prefill_runtime, 'context')
    decode_sol = get_sol_latencies(decode_model, database, decode_runtime, 'generation')
    
    # Calculate utilizations
    prefill_utilization = calculate_op_utilizations(prefill_breakdown, prefill_sol)
    decode_utilization = calculate_op_utilizations(decode_breakdown, decode_sol)
    
    # Get detailed operation information
    prefill_op_details = get_op_details(prefill_model, database, prefill_runtime, 'context')
    decode_op_details = get_op_details(decode_model, database, decode_runtime, 'generation')
    
    # Update with utilization based on actual times
    prefill_op_details = update_op_details_with_utilization(prefill_op_details, prefill_breakdown)
    decode_op_details = update_op_details_with_utilization(decode_op_details, decode_breakdown)
    
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
    
    # Calculate system overhead
    prefill_ops_time = sum(prefill_breakdown.values()) if prefill_breakdown else 0
    decode_ops_time = sum(decode_breakdown.values()) if decode_breakdown else 0
    
    prefill_overhead = ttft - prefill_ops_time if ttft > prefill_ops_time else 0
    decode_overhead = tpot - decode_ops_time if tpot > decode_ops_time else 0
    
    # Add system overhead to breakdowns
    if prefill_overhead > 0:
        prefill_breakdown['system_overhead'] = prefill_overhead
    if decode_overhead > 0:
        decode_breakdown['system_overhead'] = decode_overhead
    
    return {
        'type': 'disaggregated',
        'summary_df': combined_df,
        'prefill_df': prefill_df,
        'decode_df': decode_df,
        'prefill_throughput': prefill_throughput,
        'decode_throughput': decode_throughput,
        'bottleneck': combined_data['bottleneck'],
        'memory_info': memory_info,
        'prefill_breakdown': prefill_breakdown,
        'decode_breakdown': decode_breakdown,
        'prefill_sol': prefill_sol,
        'decode_sol': decode_sol,
        'prefill_utilization': prefill_utilization,
        'decode_utilization': decode_utilization,
        'prefill_op_details': prefill_op_details,
        'decode_op_details': decode_op_details,
        'prefill_total_time': ttft,  # Use actual TTFT
        'decode_total_time': tpot,   # Use actual TPOT
        'prefill_model': prefill_model,
        'decode_model': decode_model,
        'sol_mode': use_sol_mode,
        'system': system,
        'prefill_overhead': prefill_overhead,
        'decode_overhead': decode_overhead
    }


def calculate_utilization(op_time_ms: float, total_time_ms: float, 
                         system_spec: Dict, op_type: str, 
                         model_config: Any, batch_size: int, seq_len: int) -> float:
    """Calculate utilization rate for an operation."""
    if total_time_ms == 0 or op_time_ms == 0:
        return 0.0
    
    # Calculate theoretical peak performance based on op type
    if 'gemm' in op_type.lower():
        # GEMM ops are compute bound
        if 'fp8' in str(model_config.gemm_quant_mode):
            peak_flops = system_spec['gpu']['fp8_tc_flops']
        else:
            peak_flops = system_spec['gpu']['float16_tc_flops']
        
        # Rough estimate of FLOPs for GEMM (this is simplified)
        # Actual calculation would need matrix dimensions
        utilization = min((op_time_ms / total_time_ms) * 100, 100)
    elif 'attention' in op_type.lower():
        # Attention ops are memory bandwidth bound
        mem_bw = system_spec['gpu']['mem_bw']
        utilization = min((op_time_ms / total_time_ms) * 100, 100)
    else:
        # Default utilization calculation
        utilization = (op_time_ms / total_time_ms) * 100
    
    return utilization


def print_results(result: Dict[str, Any], verbose: bool = False):
    """Print evaluation results with detailed breakdowns."""
    
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
        if 'batch_size' in result:
            print(f"  Batch Size: {result['batch_size']} (per-GPU: {result['effective_batch_size']})")
        else:
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
    
    # Memory consumption details
    if 'memory_info' in result:
        print_memory_info(result)
    
    # Operation breakdown
    if 'context_latency' in result or 'prefill_breakdown' in result:
        print_operation_breakdown(result)
    
    if verbose:
        print("\nFull Summary DataFrame:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.to_string(index=False))
    
    print("="*80)


def print_memory_info(result: Dict[str, Any]):
    """Print memory consumption details."""
    memory_info = result.get('memory_info', {})
    
    print("\n" + "-"*60)
    print("MEMORY CONSUMPTION")
    print("-"*60)
    
    if not memory_info:
        print("  No memory information available")
        return
    
    if 'prefill' in memory_info and 'decode' in memory_info:
        # Disaggregated memory info
        print("\nPrefill Worker Memory:")
        for key, value in memory_info['prefill'].items():
            print(f"  {key:30}: {value:>10.3f} GB")
        
        print("\nDecode Worker Memory:")
        for key, value in memory_info['decode'].items():
            print(f"  {key:30}: {value:>10.3f} GB")
        
        print("\nTotal System Memory:")
        p_total = memory_info['prefill'].get('total', 0)
        d_total = memory_info['decode'].get('total', 0)
        p_workers = memory_info.get('prefill_workers', 1)
        d_workers = memory_info.get('decode_workers', 1)
        total_mem = p_total * p_workers + d_total * d_workers
        print(f"  Total GPU Memory Used: {total_mem:.2f} GB")
    else:
        # Unified memory info
        # Check if we have DP info to scale memory
        num_gpus = result.get('dp', 1) if 'dp' in result else 1
        if result['type'] == 'unified' and num_gpus > 1:
            print(f"  Per-GPU Memory:")
        for key, value in memory_info.items():
            if key != 'oom_status':
                print(f"    {key:28}: {value:>10.3f} GB")
        
        if num_gpus > 1:
            total_memory = memory_info.get('total', 0) * num_gpus
            print(f"\n  Total System Memory (all {num_gpus} GPUs): {total_memory:>10.3f} GB")
        
        if 'oom_status' in memory_info:
            print(f"\n  OOM Status: {'Yes' if memory_info['oom_status'] else 'No'}")


def print_operation_breakdown(result: Dict[str, Any]):
    """Print detailed operation breakdown with utilization."""
    print("\n" + "-"*60)
    print("OPERATION BREAKDOWN & UTILIZATION")
    print("-"*60)
    
    if result['type'] == 'unified':
        # Unified breakdown
        if 'context_latency' in result:
            # Enhanced table with FLOPs and memory
            if 'context_op_details' in result:
                print_enhanced_ops_table(
                    result['context_latency'],
                    result['context_op_details'],
                    result.get('context_total_time', sum(result['context_latency'].values())),
                    'context',
                    result.get('model'),
                    result.get('sol_mode', False)
                )
        
        if 'generation_latency' in result:
            # Enhanced table with FLOPs and memory
            if 'generation_op_details' in result:
                print_enhanced_ops_table(
                    result['generation_latency'],
                    result['generation_op_details'],
                    result.get('generation_total_time', sum(result['generation_latency'].values())),
                    'generation',
                    result.get('model'),
                    result.get('sol_mode', False)
                )
    else:
        # Disaggregated breakdown
        if 'prefill_breakdown' in result:
            # Enhanced table with FLOPs and memory
            if 'prefill_op_details' in result:
                print_enhanced_ops_table(
                    result['prefill_breakdown'],
                    result['prefill_op_details'],
                    result.get('prefill_total_time', sum(result['prefill_breakdown'].values())),
                    'prefill',
                    result.get('prefill_model'),
                    result.get('sol_mode', False)
                )
        
        if 'decode_breakdown' in result:
            # Enhanced table with FLOPs and memory
            if 'decode_op_details' in result:
                print_enhanced_ops_table(
                    result['decode_breakdown'],
                    result['decode_op_details'],
                    result.get('decode_total_time', sum(result['decode_breakdown'].values())),
                    'decode',
                    result.get('decode_model'),
                    result.get('sol_mode', False)
                )


def print_ops_table_with_utilization(ops_dict: Dict[str, float], sol_dict: Dict[str, float], 
                                     util_dict: Dict[str, float], total_time: float, sol_mode: bool = False):
    """Print operations table with exact utilization calculations."""
    if not ops_dict:
        print("  No operation data available")
        return
    
    # Separate system overhead from operations
    system_overhead = ops_dict.get('system_overhead', 0)
    ops_without_overhead = {k: v for k, v in ops_dict.items() if k != 'system_overhead'}
    
    if sol_mode:
        # In SOL mode, actual time IS the SOL time
        print(f"  {'Operation':<30} {'SOL(ms)':>10} {'%Time':>8}")
        print("  " + "-"*48)
    else:
        print(f"  {'Operation':<30} {'Actual(ms)':>10} {'SOL(ms)':>10} {'%Time':>8} {'Util%':>8}")
        print("  " + "-"*76)
    
    # Sort operations by time (descending) - exclude system overhead for now
    sorted_ops = sorted(ops_without_overhead.items(), key=lambda x: x[1], reverse=True)
    
    for op_name, op_time in sorted_ops:
        if op_time > 0.001:  # Only show ops > 0.001ms
            percentage = (op_time / total_time * 100) if total_time > 0 else 0
            
            # Special formatting for communication ops
            if 'ar_' in op_name or 'allreduce' in op_name.lower():
                op_display = f"[Comm] {op_name}"
            else:
                op_display = op_name
            
            if sol_mode:
                # In SOL mode, just show the SOL time
                print(f"  {op_display:<30} {op_time:>10.3f} {percentage:>7.1f}%")
            else:
                sol_time = sol_dict.get(op_name, 0)
                utilization = util_dict.get(op_name, 0)
                print(f"  {op_display:<30} {op_time:>10.3f} {sol_time:>10.3f} {percentage:>7.1f}% {utilization:>7.1f}%")
    
    # Add system overhead if present
    if system_overhead > 0:
        percentage = (system_overhead / total_time * 100) if total_time > 0 else 0
        if sol_mode:
            print(f"  {'[System Overhead]':<30} {system_overhead:>10.3f} {percentage:>7.1f}%")
        else:
            print(f"  {'[System Overhead]':<30} {system_overhead:>10.3f} {'N/A':>10} {percentage:>7.1f}% {'N/A':>7}")
    
    if sol_mode:
        print("  " + "-"*48)
        ops_sum = sum(ops_without_overhead.values())
        print(f"  {'Operations Total':<30} {ops_sum:>10.3f} {(ops_sum/total_time*100 if total_time > 0 else 0):>7.1f}%")
        if system_overhead > 0:
            print(f"  {'+ System Overhead':<30} {system_overhead:>10.3f} {(system_overhead/total_time*100 if total_time > 0 else 0):>7.1f}%")
        print(f"  {'TOTAL (TTFT/TPOT)':<30} {total_time:>10.3f} {100.0:>7.1f}%")
    else:
        print("  " + "-"*76)
        # Calculate overall utilization
        total_sol = sum(sol_dict.values()) if sol_dict else 0
        overall_util = (total_sol / total_time * 100) if total_time > 0 else 0
        print(f"  {'TOTAL':<30} {total_time:>10.3f} {total_sol:>10.3f} {100.0:>7.1f}% {overall_util:>7.1f}%")


def print_enhanced_ops_table(ops_dict: Dict[str, float], ops_details: Dict[str, Dict], 
                             total_time: float, phase: str, model=None, sol_mode=False):
    """Print enhanced operations table with FLOPs, memory, and separate utilization rates."""
    if not ops_dict:
        print(f"  No {phase} operation data available")
        return
    
    # Get number of layers if available to show per-layer metrics
    num_layers = getattr(model, '_num_layers', 1) if model else 1
    
    print(f"\n  {phase.capitalize()} Operations Breakdown:")
    if sol_mode:
        # For SOL mode, we show theoretical peak performance required/achieved
        print(f"  {'Operation':<28} {'Calls':>6} {'SOL Time/L(us)':>14} {'Total Time(us)':>14} {'%Time':>7} {'FLOPs/L':>10} {'Mem(MB)/L':>10} {'Achieved TFLOPS':>15} {'Achieved GB/s':>13} {'Shape':<40}")
        print("  " + "-"*180)
    else:
        print(f"  {'Operation':<28} {'Calls':>6} {'Time/L(ms)':>10} {'Total Time(ms)':>14} {'%Time':>7} {'FLOPs/L':>10} {'Mem(MB)/L':>10} {'Ach.TFLOPS':>11} {'Ach.GB/s':>10} {'CompUtil%':>10} {'MemUtil%':>9} {'Shape':<35}")
        print("  " + "-"*216)
    
    # Separate system overhead from operations
    system_overhead = ops_dict.get('system_overhead', 0)
    ops_without_overhead = {k: v for k, v in ops_dict.items() if k != 'system_overhead'}
    
    # Sort operations by time (descending) - exclude system overhead for now
    sorted_ops = sorted(ops_without_overhead.items(), key=lambda x: x[1], reverse=True)
    
    total_flops = 0
    total_mem_mb = 0
    
    for op_name, op_time in sorted_ops:
        if op_time > 0.001:  # Only show ops > 0.001ms
            percentage = (op_time / total_time * 100) if total_time > 0 else 0
            details = ops_details.get(op_name, {})
            
            shape = details.get('shape', 'N/A')
            flops = details.get('flops', 0)
            mem_bytes = details.get('memory_bytes', 0)
            
            # Calculate per-layer metrics and invoke times for ALL operations
            # Most operations happen per layer, except logits and embedding
            if any(x in op_name.lower() for x in ['logits_gemm', 'embedding']):
                # These operations happen only once (not per layer)
                invoke_times = 1
                per_layer_time = op_time
                per_layer_flops = flops
                per_layer_mem = mem_bytes
            else:
                # All other ops (including router_gemm, dispatch) happen per layer
                invoke_times = num_layers
                per_layer_time = op_time / num_layers if num_layers > 1 else op_time
                per_layer_flops = flops / num_layers if num_layers > 1 else flops
                per_layer_mem = mem_bytes / num_layers if num_layers > 1 else mem_bytes
            
            # Calculate achieved performance
            # For operations parallelized across GPUs, the per-GPU work is divided
            # We show the per-GPU performance requirement
            if sol_mode:
                # In SOL mode, the achieved performance is what's required to meet SOL time
                # Convert per_layer_time from ms to seconds for TFLOPS/GB/s calculation
                per_layer_time_us = per_layer_time * 1000  # Convert ms to us for display
                achieved_tflops = (per_layer_flops / 1e12) / (per_layer_time / 1000) if per_layer_time > 0 else 0
                achieved_gb_s = (per_layer_mem / 1e9) / (per_layer_time / 1000) if per_layer_time > 0 else 0
            else:
                # In actual mode, show what was actually achieved
                achieved_tflops = (per_layer_flops / 1e12) / (per_layer_time / 1000) if per_layer_time > 0 else 0
                achieved_gb_s = (per_layer_mem / 1e9) / (per_layer_time / 1000) if per_layer_time > 0 else 0
            
            # Recalculate utilization based on SOL values
            sol_time = details.get('sol_time', 0)
            sol_math = details.get('sol_math', 0) 
            sol_mem = details.get('sol_mem', 0)
            
            # For per-layer metrics, also divide SOL times by num_layers
            if not any(x in op_name.lower() for x in ['logits_gemm', 'embedding']):
                sol_time = sol_time / num_layers if num_layers > 1 else sol_time
                sol_math = sol_math / num_layers if num_layers > 1 else sol_math
                sol_mem = sol_mem / num_layers if num_layers > 1 else sol_mem
            
            # Calculate utilization
            comp_util = (sol_math / per_layer_time * 100) if per_layer_time > 0 and sol_math > 0 else 0
            mem_util = (sol_mem / per_layer_time * 100) if per_layer_time > 0 and sol_mem > 0 else 0
            
            # Format values for display - no truncation for shape
            shape_str = str(shape) if shape else 'N/A'
            
            flops_str = f"{per_layer_flops/1e9:.2f}G" if per_layer_flops > 0 else "N/A"
            mem_mb = per_layer_mem / 1e6
            mem_str = f"{mem_mb:.2f}" if per_layer_mem > 0 else "N/A"
            achieved_tflops_str = f"{achieved_tflops:.2f}" if achieved_tflops > 0 else "N/A"
            achieved_gb_s_str = f"{achieved_gb_s:.1f}" if achieved_gb_s > 0 else "N/A"
            
            total_flops += flops
            total_mem_mb += mem_bytes / 1e6
            
            # Special formatting for communication ops
            if 'ar_' in op_name or 'allreduce' in op_name.lower():
                op_display = f"[Comm] {op_name[:21]}"  # Truncate to fit
            else:
                op_display = op_name
            
            # Calculate total time for this operation
            total_op_time = op_time  # This is already the total time across all invocations
            
            if sol_mode:
                # In SOL mode, time IS the SOL time, show achieved/required performance
                # Display time in microseconds
                per_layer_time_us = per_layer_time * 1000  # Convert ms to us
                total_op_time_us = total_op_time * 1000  # Convert ms to us
                print(f"  {op_display:<28} {invoke_times:>6} {per_layer_time_us:>14.1f} {total_op_time_us:>14.1f} {percentage:>6.1f}% {flops_str:>10} {mem_str:>10} {achieved_tflops_str:>15} {achieved_gb_s_str:>13} {shape_str:<40}")
            else:
                print(f"  {op_display:<28} {invoke_times:>6} {per_layer_time:>10.3f} {total_op_time:>14.3f} {percentage:>6.1f}% {flops_str:>10} {mem_str:>10} {achieved_tflops_str:>11} {achieved_gb_s_str:>10} {comp_util:>9.1f}% {mem_util:>8.1f}% {shape_str:<35}")
    
    # Add system overhead if present
    if system_overhead > 0:
        percentage = (system_overhead / total_time * 100) if total_time > 0 else 0
        if sol_mode:
            system_overhead_us = system_overhead * 1000  # Convert ms to us
            print(f"  {'[System Overhead]':<28} {'N/A':>6} {'N/A':>14} {system_overhead_us:>14.1f} {percentage:>6.1f}% {'N/A':>10} {'N/A':>10} {'N/A':>15} {'N/A':>13} {'N/A':<40}")
        else:
            print(f"  {'[System Overhead]':<28} {'N/A':>6} {'N/A':>10} {system_overhead:>14.3f} {percentage:>6.1f}% {'N/A':>10} {'N/A':>10} {'N/A':>11} {'N/A':>10} {'N/A':>10} {'N/A':>9} {'N/A':<35}")
    
    if sol_mode:
        print("  " + "-"*180)
        # Total row
        total_flops_str = f"{total_flops/1e9:.1f}G" if total_flops > 0 else "N/A"
        ops_sum = sum(ops_without_overhead.values())
        ops_sum_us = ops_sum * 1000  # Convert ms to us
        ops_percentage = (ops_sum / total_time * 100) if total_time > 0 else 0
        
        # Calculate achieved TFLOPS and GB/s for operations total
        total_achieved_tflops = (total_flops / 1e12) / (ops_sum / 1000) if ops_sum > 0 else 0
        total_achieved_gb_s = (total_mem_mb / 1e3) / (ops_sum / 1000) if ops_sum > 0 else 0  # MB to GB, ms to s
        total_achieved_tflops_str = f"{total_achieved_tflops:.2f}" if total_achieved_tflops > 0 else ""
        total_achieved_gb_s_str = f"{total_achieved_gb_s:.1f}" if total_achieved_gb_s > 0 else ""
        
        print(f"  {'Operations Total':<28} {'':<6} {'':<14} {ops_sum_us:>14.1f} {ops_percentage:>6.1f}% {total_flops_str:>10} {total_mem_mb:>10.1f} {total_achieved_tflops_str:>15} {total_achieved_gb_s_str:>13} {'':<40}")
        if system_overhead > 0:
            total_time_us = total_time * 1000  # Convert ms to us
            print(f"  {'TOTAL (TTFT/TPOT)':<28} {'':<6} {'':<14} {total_time_us:>14.1f} {'100.0%':>7} {'':<10} {'':<10} {'':<15} {'':<13} {'':<40}")
    else:
        print("  " + "-"*216)
        # Total row (showing cumulative values)
        total_flops_str = f"{total_flops/1e9:.1f}G" if total_flops > 0 else "N/A"
        print(f"  {'TOTAL (all layers)':<28} {'':<6} {'':<10} {total_time:>14.3f} {'100.0%':>7} {total_flops_str:>10} {total_mem_mb:>10.1f} {'':<11} {'':<10} {'':<10} {'':<9} {'':<35}")


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