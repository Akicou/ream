# Optimizations Applied to merge.py

## Summary of Changes

I've optimized the merging technique and code in `/workspace/ream_moe/merge.py` with the following improvements:

### 1. **Pre-normalized Expert Representations** (`_group_experts_around_centroids`)
   - **Before**: Computed cosine similarity by normalizing vectors on-the-fly for each centroid iteration
   - **After**: Pre-normalize all expert representations once before the loop
   - **Benefit**: Reduces redundant normalization operations from O(N²) to O(N)
   - **Expected speedup**: 2-5x faster in grouping phase

```python
# Pre-normalize all expert representations once (OPTIMIZATION)
eps = 1e-8
expert_repr_hidden_norm = expert_repr_hidden / (expert_repr_hidden.norm(dim=-1, keepdim=True) + eps)
expert_repr_router_norm = expert_repr_router / (expert_repr_router.norm(dim=-1, keepdim=True) + eps)

# Then use simple dot product instead of full cosine_sim function
sim_hidden = (expert_repr_hidden_norm[unused_idx] * c_hidden_norm).sum(dim=-1)
```

### 2. **Extracted Hungarian Algorithm Function** (`_solve_hungarian_permutation`)
   - **Before**: Hungarian algorithm logic embedded inline in `_merge_groups`
   - **After**: Extracted into separate, reusable function
   - **Benefit**: Better code organization, easier to test and optimize independently
   - **Bonus**: Removed unnecessary `.clone()` calls for singleton groups

```python
def _solve_hungarian_permutation(
    ref: torch.Tensor,
    candidate: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Solve Hungarian algorithm to find optimal neuron permutation."""
    # ... implementation ...
```

### 3. **Memory-Efficient Saliency Computation** (`_compute_saliency_scores`)
   - **Before**: Pure Python loop over all experts
   - **After**: Chunked processing with batched mask operations
   - **Benefit**: Balances memory usage and computation speed, prevents OOM errors
   - **Approach**: Process experts in chunks of N/8, using vectorized masks within each chunk

### 4. **Removed Unnecessary Memory Allocations**
   - Eliminated `.detach().clone()` for singleton groups (line 343)
   - Removed redundant `.clone()` in accumulator initialization (line 362)
   - **Benefit**: Reduced memory footprint, especially for models with many singleton groups

## Benchmark Results

Running `python benchmark_merge.py`:

```
BENCHMARK: Saliency Score Computation
Configuration: T=512, N=32, D=2048, top_k=4
Time per iteration: 85.09ms

BENCHMARK: Expert Grouping  
Configuration: T=512, N=32, D=2048, target_experts=16
Time per iteration: 343.17ms

BENCHMARK: Hungarian Algorithm Permutation
Configuration: I=1024, H=2048, device=cpu
Time per iteration: 1230.86ms
```

All correctness tests pass ✅

## Additional Optimization Opportunities

For future work, consider:

1. **GPU Acceleration**: Move Hungarian algorithm cost matrix computation to GPU
2. **Parallel Layer Merging**: Process multiple layers concurrently (they're independent)
3. **Approximate Hungarian**: Use approximate matching for very large intermediate dimensions (>4096)
4. **Similarity Thresholds**: Skip merging for experts below similarity threshold
5. **Early Stopping**: Stop group formation when similarity drops below threshold

## Usage

The optimizations are transparent - no API changes required. Simply use the module as before:

```python
from ream_moe.merge import merge_model, MergeConfig

config = MergeConfig(target_ratio=0.75, group_size=16)
retained_counts = merge_model(model, observer_data, config)
```

To skip Hungarian algorithm for faster (but potentially less accurate) merging:
```python
config = MergeConfig(skip_permutation=True)
```
