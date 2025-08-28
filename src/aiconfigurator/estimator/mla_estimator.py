# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-head Latent Attention (MLA) performance estimator for DeepSeek V3.
"""

import os
from .roofline_estimator import RooflineEstimator

class MLAEstimator(RooflineEstimator):
    """Estimator for DeepSeek V3's Multi-head Latent Attention mechanism."""
    
    def generate_mla_files(self, output_dir: str):
        """Generate MLA-specific performance files for DeepSeek V3."""
        
        # Context MLA
        self._generate_context_mla_file(os.path.join(output_dir, "context_mla_perf.txt"))
        
        # Generation MLA  
        self._generate_generation_mla_file(os.path.join(output_dir, "generation_mla_perf.txt"))
        
        # MLA BMM (batch matrix multiplication)
        self._generate_mla_bmm_file(os.path.join(output_dir, "mla_bmm_perf.txt"))
    
    def _estimate_mla_time(self, seq_len: int, latent_dim: int, num_heads: int, 
                          batch_size: int = 1, dtype: str = 'fp16', 
                          is_context: bool = True, utilization: float = 0.85) -> float:
        """
        Estimate MLA computation time.
        
        MLA uses compressed key-value representations with latent attention.
        Key differences from standard attention:
        1. Compressed KV cache (latent_dim << head_dim * num_heads)
        2. Different computation pattern for Q@K and attention weights
        """
        
        if is_context:
            # Context phase: full sequence attention with compression
            # Approximate FLOPs: similar to standard attention but with latent compression
            latent_flops = 4 * seq_len * seq_len * latent_dim * batch_size
            projection_flops = 2 * seq_len * num_heads * latent_dim * batch_size
            total_flops = latent_flops + projection_flops
        else:
            # Generation phase: single token with compressed KV cache  
            latent_flops = 4 * seq_len * latent_dim * batch_size
            projection_flops = 2 * num_heads * latent_dim * batch_size
            total_flops = latent_flops + projection_flops
            
        # Get compute throughput
        if dtype == 'fp8':
            peak_flops = self.fp8_flops
        else:
            peak_flops = self.fp16_flops
            
        compute_time = total_flops / (peak_flops * utilization)
        
        # Memory traffic (compressed compared to standard attention)
        dtype_bytes = {'fp16': 2, 'fp8': 1}.get(dtype, 2)
        if is_context:
            # Latent KV cache is much smaller than full KV cache
            memory_traffic = batch_size * seq_len * latent_dim * 2 * dtype_bytes  # compressed K,V
        else:
            memory_traffic = batch_size * seq_len * latent_dim * dtype_bytes  # compressed cache access
            
        memory_time = memory_traffic / (self.mem_bw * self.mem_bw_scaling)
        
        return max(compute_time, memory_time) + self.mem_latency
    
    def _generate_context_mla_file(self, filename: str):
        """Generate context MLA performance file."""
        with open(filename, 'w') as f:
            # CSV header: mla_dtype,kv_cache_dtype,batch_size,isl,tp_size,latency
            f.write("mla_dtype,kv_cache_dtype,batch_size,isl,tp_size,latency\n")
            
            # DeepSeek V3 MLA configuration
            seq_lens = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768]
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
            tp_sizes = [1, 2, 4, 8, 16, 32, 64, 72]
            
            for seq_len in seq_lens:
                for batch_size in batch_sizes:
                    for tp_size in tp_sizes:
                        if tp_size <= 72:  # NVL72 limit
                            for mla_dtype in ['float16', 'fp8']:
                                for kv_cache_dtype in ['float16', 'fp8']:
                                    time_sec = self._estimate_mla_time(
                                        seq_len, 1536, 128, batch_size, 
                                        'fp16' if mla_dtype == 'float16' else 'fp8', 
                                        is_context=True
                                    )
                                    time_ms = time_sec * 1000
                                    f.write(f"{mla_dtype},{kv_cache_dtype},{batch_size},{seq_len},{tp_size},{time_ms:.4f}\n")
    
    def _generate_generation_mla_file(self, filename: str):
        """Generate generation MLA performance file."""
        with open(filename, 'w') as f:
            # CSV header: mla_dtype,kv_cache_dtype,batch_size,isl,tp_size,step,latency
            f.write("mla_dtype,kv_cache_dtype,batch_size,isl,tp_size,step,latency\n")
            
            seq_lens = [1, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384]  # KV cache length
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            tp_sizes = [1, 2, 4, 8, 16, 32, 64, 72]
            steps = [1, 10, 100, 500, 1000]  # Generation steps
            
            for seq_len in seq_lens:
                for batch_size in batch_sizes:
                    for tp_size in tp_sizes:
                        for step in steps:
                            if tp_size <= 72:  # NVL72 limit
                                for mla_dtype in ['float16', 'fp8']:
                                    for kv_cache_dtype in ['float16', 'fp8']:
                                        time_sec = self._estimate_mla_time(
                                            seq_len, 1536, 128, batch_size,
                                            'fp16' if mla_dtype == 'float16' else 'fp8',
                                            is_context=False
                                        )
                                        time_ms = time_sec * 1000
                                        f.write(f"{mla_dtype},{kv_cache_dtype},{batch_size},{seq_len},{tp_size},{step},{time_ms:.4f}\n")
    
    def _generate_mla_bmm_file(self, filename: str):
        """Generate MLA batch matrix multiplication performance file."""
        with open(filename, 'w') as f:
            # CSV header: bmm_dtype,num_tokens,num_heads,latency,op_name
            f.write("bmm_dtype,num_tokens,num_heads,latency,op_name\n")
            
            # BMM token counts for MLA operations
            token_counts = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
            num_heads_list = [32, 64, 96, 128]  # Multiple head configurations for proper interpolation
            op_names = ['bmm_pre', 'bmm_post']  # Before and after attention
            
            for num_tokens in token_counts:
                for num_heads in num_heads_list:
                    for op_name in op_names:
                        for bmm_dtype in ['float16', 'fp8_block']:
                            # Estimate based on typical MLA BMM operations
                            # Using simplified GEMM estimation for BMM
                            time_sec = self.estimate_gemm_time(num_tokens, 1536, 1536, 
                                                             'fp16' if bmm_dtype == 'float16' else 'fp8')
                            time_ms = time_sec * 1000
                            f.write(f"{bmm_dtype},{num_tokens},{num_heads},{time_ms:.4f},{op_name}\n")