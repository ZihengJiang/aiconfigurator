# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import yaml
import os
import math
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class RooflineEstimator:
    """
    Roofline-based performance estimator for GEMM and communication operations.
    Uses hardware specifications to estimate performance without real data collection.
    """
    
    def __init__(self, system_config_path: str):
        """
        Initialize the roofline estimator with system configuration.
        
        Args:
            system_config_path: Path to system YAML configuration file
        """
        with open(system_config_path, 'r') as f:
            self.system_config = yaml.safe_load(f)
        
        # Extract hardware specs
        gpu = self.system_config['gpu']
        self.mem_bw = gpu['mem_bw']  # bytes/s
        self.mem_bw_scaling = gpu.get('mem_bw_empirical_scaling_factor', 1.0)
        self.mem_latency = gpu.get('mem_empirical_constant_latency', 0.0)
        self.fp16_flops = gpu['float16_tc_flops']  # ops/s
        self.fp8_flops = gpu['fp8_tc_flops']  # ops/s
        self.int8_flops = gpu['int8_tc_flops']  # ops/s
        
        # Network specs
        node = self.system_config['node']
        self.intra_node_bw = node['intra_node_bw']  # bytes/s per GPU
        self.inter_node_bw = node['inter_node_bw']  # bytes/s per GPU
        self.p2p_latency = node.get('p2p_latency', 0.00001)  # seconds
        
    def estimate_gemm_time(self, M: int, N: int, K: int, dtype: str = 'fp16', 
                          utilization: float = 0.85) -> float:
        """
        Estimate GEMM execution time using roofline model.
        
        Args:
            M, N, K: Matrix dimensions for M×K @ K×N = M×N
            dtype: Data type ('fp16', 'fp8', 'int8')
            utilization: Hardware utilization factor (0-1)
            
        Returns:
            Estimated execution time in seconds
        """
        # Calculate FLOPs
        flops = 2 * M * N * K  # Each element requires one multiply-add
        
        # Get peak compute throughput
        if dtype == 'fp8':
            peak_flops = self.fp8_flops
        elif dtype == 'int8':
            peak_flops = self.int8_flops
        else:  # fp16 default
            peak_flops = self.fp16_flops
            
        # Calculate memory traffic (bytes)
        dtype_bytes = {'fp16': 2, 'fp8': 1, 'int8': 1}.get(dtype, 2)
        memory_traffic = (M * K + K * N + M * N) * dtype_bytes
        
        # Roofline: limited by either compute or memory bandwidth
        compute_time = flops / (peak_flops * utilization)
        memory_time = memory_traffic / (self.mem_bw * self.mem_bw_scaling)
        
        # Add constant latency overhead
        estimated_time = max(compute_time, memory_time) + self.mem_latency
        
        return estimated_time
    
    def estimate_allreduce_time(self, message_size_bytes: int, num_gpus: int, 
                               utilization: float = 0.7) -> float:
        """
        Estimate AllReduce communication time.
        
        Args:
            message_size_bytes: Size of message to reduce
            num_gpus: Number of participating GPUs
            utilization: Network utilization factor
            
        Returns:
            Estimated execution time in seconds
        """
        if num_gpus <= 1:
            return 0.0
            
        # Ring AllReduce algorithm: 2 * (P-1) / P * message_size / bandwidth
        # Where P is number of participants
        ring_factor = 2 * (num_gpus - 1) / num_gpus
        
        # Use intra-node bandwidth (assuming single node or optimized multi-node)
        bandwidth = self.intra_node_bw * utilization
        
        transfer_time = ring_factor * message_size_bytes / bandwidth
        
        # Add latency overhead (scales with number of hops)
        latency_overhead = self.p2p_latency * math.log2(num_gpus)
        
        return transfer_time + latency_overhead
    
    def estimate_attention_time(self, seq_len: int, num_heads: int, head_dim: int,
                               batch_size: int = 1, dtype: str = 'fp16',
                               is_context: bool = True, utilization: float = 0.8) -> float:
        """
        Estimate attention computation time.
        
        Args:
            seq_len: Sequence length
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            batch_size: Batch size
            dtype: Data type
            is_context: True for context (prefill), False for generation
            utilization: Hardware utilization
            
        Returns:
            Estimated execution time in seconds
        """
        if is_context:
            # Context attention: O(seq_len^2) complexity
            flops_per_head = 4 * seq_len * seq_len * head_dim * batch_size
        else:
            # Generation attention: O(seq_len) complexity for single token
            flops_per_head = 4 * seq_len * head_dim * batch_size
            
        total_flops = flops_per_head * num_heads
        
        # Get compute throughput
        if dtype == 'fp8':
            peak_flops = self.fp8_flops
        else:
            peak_flops = self.fp16_flops
            
        compute_time = total_flops / (peak_flops * utilization)
        
        # Memory traffic estimation (simplified)
        dtype_bytes = {'fp16': 2, 'fp8': 1}.get(dtype, 2)
        if is_context:
            memory_traffic = batch_size * num_heads * seq_len * head_dim * 3 * dtype_bytes  # Q, K, V
        else:
            memory_traffic = batch_size * num_heads * seq_len * head_dim * 2 * dtype_bytes  # K, V cache access
            
        memory_time = memory_traffic / (self.mem_bw * self.mem_bw_scaling)
        
        return max(compute_time, memory_time) + self.mem_latency
    
    def estimate_moe_time(self, hidden_size: int, intermediate_size: int, 
                         num_experts: int, top_k: int, batch_size: int = 1,
                         dtype: str = 'fp16', utilization: float = 0.75) -> float:
        """
        Estimate MoE (Mixture of Experts) computation time.
        
        Args:
            hidden_size: Hidden dimension
            intermediate_size: Intermediate FFN dimension  
            num_experts: Total number of experts
            top_k: Number of experts to activate
            batch_size: Batch size
            dtype: Data type
            utilization: Hardware utilization
            
        Returns:
            Estimated execution time in seconds
        """
        # Router computation
        router_flops = 2 * batch_size * hidden_size * num_experts
        
        # Expert computation (only top_k experts are activated)
        # Each expert: 2 FFN layers
        expert_flops = 2 * batch_size * top_k * (
            2 * hidden_size * intermediate_size +  # up projection
            2 * intermediate_size * hidden_size    # down projection  
        )
        
        total_flops = router_flops + expert_flops
        
        # Get compute throughput
        if dtype == 'fp8':
            peak_flops = self.fp8_flops
        elif dtype == 'int8':
            peak_flops = self.int8_flops
        else:
            peak_flops = self.fp16_flops
            
        compute_time = total_flops / (peak_flops * utilization)
        
        # Memory traffic (simplified - includes dispatch overhead)
        dtype_bytes = {'fp16': 2, 'fp8': 1, 'int8': 1}.get(dtype, 2)
        memory_traffic = batch_size * (
            hidden_size * num_experts * dtype_bytes +  # router weights
            top_k * intermediate_size * hidden_size * 2 * dtype_bytes  # expert weights
        )
        
        memory_time = memory_traffic / (self.mem_bw * self.mem_bw_scaling)
        
        return max(compute_time, memory_time) + self.mem_latency

    def generate_performance_files(self, output_dir: str):
        """
        Generate performance data files with estimated values.
        
        Args:
            output_dir: Directory to write performance files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate GEMM performance data
        self._generate_gemm_perf_file(os.path.join(output_dir, "gemm_perf.txt"))
        
        # Generate NCCL performance data  
        self._generate_nccl_perf_file(os.path.join(output_dir, "nccl_perf.txt"))
        
        # Generate attention performance data
        self._generate_attention_perf_files(output_dir)
        
        # Generate MoE performance data
        self._generate_moe_perf_file(os.path.join(output_dir, "moe_perf.txt"))
        
        # Generate custom allreduce data
        self._generate_custom_allreduce_file(os.path.join(output_dir, "custom_allreduce_perf.txt"))
        
        logger.info(f"Generated performance estimation files in {output_dir}")
    
    def _generate_gemm_perf_file(self, filename: str):
        """Generate GEMM performance estimation file."""
        with open(filename, 'w') as f:
            # Write CSV header
            f.write("gemm_dtype,m,n,k,latency\n")
            
            # Common GEMM shapes for LLM inference
            shapes = [
                # Small shapes
                (1, 4096, 4096), (1, 8192, 4096), (1, 11008, 4096),
                (1, 4096, 11008), (1, 14336, 4096), (1, 4096, 14336),
                # Medium batch sizes
                (16, 4096, 4096), (32, 4096, 4096), (64, 4096, 4096),
                (128, 4096, 4096), (256, 4096, 4096), (512, 4096, 4096),
                # Large shapes for prefill
                (1024, 4096, 4096), (2048, 4096, 4096), (4096, 4096, 4096),
                # DeepSeek V3 specific shapes  
                (1, 7168, 7168), (1, 18432, 7168), (1, 7168, 18432),
                (32, 7168, 7168), (128, 7168, 7168), (512, 7168, 7168),
                (1, 2048, 7168), (16, 2048, 7168), (32, 2048, 7168),
            ]
            
            # Map our dtype names to common enum names
            dtype_mapping = {
                'fp16': 'float16',
                'fp8': 'fp8_block',  # Use fp8_block to match the config
                'int8': 'int8_wo'
            }
            
            for M, N, K in shapes:
                for dtype in ['fp16', 'fp8', 'int8']:
                    time_sec = self.estimate_gemm_time(M, N, K, dtype)
                    time_ms = time_sec * 1000
                    gemm_dtype = dtype_mapping[dtype]
                    f.write(f"{gemm_dtype},{M},{N},{K},{time_ms:.4f}\n")
    
    def _generate_nccl_perf_file(self, filename: str):
        """Generate NCCL AllReduce performance estimation file."""
        with open(filename, 'w') as f:
            # Write CSV header
            f.write("nccl_dtype,num_gpus,message_size,op_name,latency\n")
            
            # Message sizes (bytes)
            sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
            gpu_counts = [2, 4, 8, 16, 32, 72]
            
            for size_bytes in sizes:
                for num_gpus in gpu_counts:
                    if num_gpus <= 72:  # NVL72 limit
                        time_sec = self.estimate_allreduce_time(size_bytes, num_gpus)
                        time_us = time_sec * 1e6
                        
                        # Write in CSV format (nccl_dtype,num_gpus,message_size,op_name,latency)
                        f.write(f"half,{num_gpus},{size_bytes},allreduce,{time_us:.4f}\n")
    
    def _generate_attention_perf_files(self, output_dir: str):
        """Generate attention performance files for context and generation."""
        
        # Context attention
        with open(os.path.join(output_dir, "context_attention_perf.txt"), 'w') as f:
            # CSV header: attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,latency
            f.write("attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,latency\n")
            
            seq_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
            head_configs = [(32, 32), (40, 40), (64, 8), (128, 128)]  # (num_heads, num_kv_heads)
            batch_sizes = [1, 4, 8, 16, 32]
            
            for seq_len in seq_lens:
                for num_heads, num_kv_heads in head_configs:
                    for batch_size in batch_sizes:
                        for attn_dtype in ['float16', 'fp8']:
                            for kv_cache_dtype in ['float16', 'fp8']:
                                time_sec = self.estimate_attention_time(
                                    seq_len, num_heads, 128, batch_size, 
                                    'fp16' if attn_dtype == 'float16' else 'fp8', 
                                    is_context=True
                                )
                                time_ms = time_sec * 1000
                                f.write(f"{attn_dtype},{kv_cache_dtype},{batch_size},{seq_len},{num_heads},{num_kv_heads},{time_ms:.4f}\n")
        
        # Generation attention  
        with open(os.path.join(output_dir, "generation_attention_perf.txt"), 'w') as f:
            # CSV header: attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,step,latency
            f.write("attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,step,latency\n")
            
            seq_lens = [1, 512, 1024, 2048, 4096, 8192]  # KV cache length
            steps = [1, 10, 100]  # Generation steps
            
            for seq_len in seq_lens:
                for num_heads, num_kv_heads in head_configs:
                    for batch_size in [1, 16, 32, 64, 128, 256, 512]:
                        for step in steps:
                            for attn_dtype in ['float16', 'fp8']:
                                for kv_cache_dtype in ['float16', 'fp8']:
                                    time_sec = self.estimate_attention_time(
                                        seq_len, num_heads, 128, batch_size,
                                        'fp16' if attn_dtype == 'float16' else 'fp8',
                                        is_context=False
                                    )
                                    time_ms = time_sec * 1000
                                    f.write(f"{attn_dtype},{kv_cache_dtype},{batch_size},{seq_len},{num_heads},{num_kv_heads},{step},{time_ms:.4f}\n")
    
    def _generate_moe_perf_file(self, filename: str):
        """Generate MoE performance estimation file.""" 
        with open(filename, 'w') as f:
            # CSV header: moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,kernel_source,latency
            f.write("moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,kernel_source,latency\n")
            
            # DeepSeek V3 MoE configs
            configs = [
                (7168, 2048, 256, 8),  # DeepSeek V3 standard config
                (7168, 18432, 3, 3),   # DeepSeek V3 dense layers
            ]
            
            batch_sizes = [1, 16, 32, 64, 128, 256]
            moe_tp_sizes = [1, 2, 4, 8, 16, 32]
            moe_ep_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            
            dtype_mapping = {
                'fp16': 'float16',
                'fp8': 'fp8_block',
                'int8': 'w4afp8'  # Map to MoE quantization mode
            }
            
            for hidden_size, inter_size, num_experts, top_k in configs:
                for num_tokens in batch_sizes:
                    for moe_tp_size in moe_tp_sizes:
                        for moe_ep_size in moe_ep_sizes:
                            if moe_tp_size * moe_ep_size <= 72 and moe_ep_size <= num_experts:  # NVL72 limit
                                for dtype in ['fp16', 'fp8', 'int8']:
                                    time_sec = self.estimate_moe_time(
                                        hidden_size, inter_size, num_experts, top_k, num_tokens, dtype
                                    )
                                    time_ms = time_sec * 1000
                                    moe_dtype = dtype_mapping[dtype]
                                    
                                    # distribution and kernel_source are placeholder values
                                    f.write(f"{moe_dtype},{num_tokens},{hidden_size},{inter_size},{top_k},{num_experts},{moe_tp_size},{moe_ep_size},balanced,moe_torch_flow,{time_ms:.4f}\n")
    
    def _generate_custom_allreduce_file(self, filename: str):
        """Generate custom allreduce performance file."""
        with open(filename, 'w') as f:
            # Format: dtype,tp_size,message_size,allreduce_strategy,layer_name,latency
            
            sizes = [4096, 16384, 65536, 262144, 1048576, 4194304]
            gpu_counts = [2, 4, 8, 16, 32, 72]
            strategies = ['NCCL', 'ONESHOT', 'TWOSHOT']
            
            for size_bytes in sizes:
                for num_gpus in gpu_counts:
                    if num_gpus <= 72:
                        base_time = self.estimate_allreduce_time(size_bytes, num_gpus)
                        
                        for strategy in strategies:
                            # Apply strategy-specific scaling factors
                            if strategy == 'ONESHOT':
                                time_sec = base_time * 0.8  # Slightly faster
                            elif strategy == 'TWOSHOT':
                                time_sec = base_time * 1.2  # Slightly slower
                            else:  # NCCL
                                time_sec = base_time
                                
                            time_us = time_sec * 1e6
                            # Format: dtype,tp_size,message_size,allreduce_strategy,layer_name,latency
                            f.write(f"half,{num_gpus},{size_bytes},{strategy},ar_layer,{time_us:.4f}\n")