# AIConfigurator vs Breakdown Analysis Discrepancy

## Summary of Issues

### 1. SOL Time Paradox
**Problem**: SOL (Speed of Light) time should be the theoretical minimum (fastest possible), but our breakdown shows:
- SOL TTFT: 3795.61ms 
- Actual TTFT: 452.79ms

This is backwards - SOL should be ≤ Actual, not 8.4x slower!

### 2. Root Causes

#### A. Batch Size Misunderstanding
AIConfigurator's deployment model:
- **64 GPUs total = 8 replicas × 8 GPUs/replica**
- **Global batch 128 = distributed across 8 replicas**
- **Each replica processes 16 different requests**

Our evaluation:
- Single 8-GPU instance trying to process 128 requests
- With DP=8, this means 128×10240 tokens = 1,310,720 tokens
- This is 8x more work than what each replica actually does

#### B. SOL Calculation Issue in SDK
The SDK's SOL calculation appears to be:
1. Not accounting for parallelism benefits correctly
2. Computing time for full batch on each GPU (replicated work)
3. Missing optimizations that AIConfigurator's model includes

#### C. Memory Calculation
- Our calculation: 291GB per GPU × 8 = 2331GB (OOM)
- Reality: Weights are shared within DP group
- Actual memory: ~96GB weights + smaller activations/KV per GPU

## Correct Interpretation

### What AIConfigurator Actually Measures
```
Per Replica (8 GPUs):
- Batch size: 16 requests
- Tokens: 16 × 10240 = 163,840 
- TTFT: 452.79ms (actual, with overheads)
- TPOT: 48.34ms
- Memory: Fits in 96GB HBM3e per GPU

Aggregate (8 replicas):
- Total batch: 128 requests
- Total throughput: 8 × per-replica
- Tokens/s/GPU: 2581.92 (same, work is replicated)
```

### What Our Breakdown Shows
```
Single Instance (8 GPUs):
- Batch size: 128 requests (8x too much)
- Tokens: 1,310,720 (8x too much)
- TTFT: 3795.61ms (SOL theoretical)
- Memory: 2331GB (incorrect due to DP replication)
```

## The Real Issue

The SDK's SOL mode for GB200 NVL72 seems to have issues:

1. **SOL should be optimistic** (faster than actual), but it's pessimistic
2. **Parallelism benefits** aren't properly accounted for
3. **Memory calculation** doesn't account for weight sharing in DP

## Recommendations

1. **Fix SOL Calculation**: The SDK needs calibration for GB200 NVL72
2. **Account for Replicas**: Evaluation should match deployment model
3. **Memory Model**: Fix DP weight sharing calculation
4. **Add Efficiency Factors**: SOL → Actual needs realistic efficiency (10-20%)

## Actual vs SOL Performance

Based on the data:
- **Actual TTFT**: 452.79ms
- **SOL TTFT**: Should be ~50-100ms (10-20% of actual)
- **Current SOL**: 3795.61ms (wrong!)

The SOL calculation is off by approximately 75x!