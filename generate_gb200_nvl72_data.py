#!/usr/bin/env python3
"""
Script to generate performance data for GB200 NVL72 system using roofline estimation.
"""

import os
import sys

# Add the src directory to the Python path so we can import the estimator modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from aiconfigurator.estimator.roofline_estimator import RooflineEstimator
from aiconfigurator.estimator.mla_estimator import MLAEstimator

def main():
    # Paths
    system_config_path = "src/aiconfigurator/systems/gb200_nvl72.yaml"
    base_output_dir = "src/aiconfigurator/systems/data/gb200_nvl72"
    
    # TensorRT-LLM version directories
    trtllm_versions = ["0.20.0", "1.0.0rc3"]
    nccl_versions = ["2.23"]
    
    print("Generating GB200 NVL72 performance data...")
    
    # Initialize estimators
    roofline_estimator = RooflineEstimator(system_config_path)
    mla_estimator = MLAEstimator(system_config_path)
    
    # Generate TensorRT-LLM performance data for each version
    for version in trtllm_versions:
        print(f"Generating TensorRT-LLM {version} performance data...")
        output_dir = os.path.join(base_output_dir, "trtllm", version)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all performance files using roofline estimator
        roofline_estimator.generate_performance_files(output_dir)
        
        print(f"TensorRT-LLM {version} performance files generated in {output_dir}")
    
    # Generate NCCL performance data
    for version in nccl_versions:
        print(f"Generating NCCL {version} performance data...")
        output_dir = os.path.join(base_output_dir, "nccl", version)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate NCCL performance file
        roofline_estimator._generate_nccl_perf_file(os.path.join(output_dir, "nccl_perf.txt"))
        
        print(f"NCCL {version} performance files generated in {output_dir}")
    
    print("All GB200 NVL72 performance data generated successfully!")
    print("\nGenerated files:")
    for root, dirs, files in os.walk(base_output_dir):
        for file in files:
            if file.endswith(".txt"):
                print(f"  {os.path.join(root, file)}")

if __name__ == "__main__":
    main()