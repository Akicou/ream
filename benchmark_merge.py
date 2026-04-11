"""
Benchmark script to test and measure optimization improvements in merge.py
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any

# Import the optimized functions
from ream_moe.merge import (
    _compute_saliency_scores,
    _group_experts_around_centroids,
    _merge_groups,
    _solve_hungarian_permutation,
    MergeConfig,
)


def benchmark_saliency_computation():
    """Benchmark the vectorized saliency score computation."""
    print("\n" + "="*60)
    print("BENCHMARK: Saliency Score Computation")
    print("="*60)
    
    # Setup test data (reduced size for memory efficiency)
    T, N, D = 512, 32, 2048  # tokens, experts, hidden_dim
    top_k = 4
    
    router_logits = torch.randn(T, N, device='cuda' if torch.cuda.is_available() else 'cpu')
    expert_outputs = torch.randn(N, T, D, device=router_logits.device)
    observer_stats = {}
    
    # Warmup
    for _ in range(3):
        _ = _compute_saliency_scores(router_logits, expert_outputs, observer_stats, "saliency_scores", top_k)
    
    # Benchmark
    iterations = 20
    start = time.time()
    for _ in range(iterations):
        saliency = _compute_saliency_scores(router_logits, expert_outputs, observer_stats, "saliency_scores", top_k)
    elapsed = time.time() - start
    
    print(f"Configuration: T={T}, N={N}, D={D}, top_k={top_k}")
    print(f"Time per iteration: {elapsed/iterations*1000:.2f}ms")
    print(f"Saliency shape: {saliency.shape}")
    print(f"Saliency stats: min={saliency.min():.4f}, max={saliency.max():.4f}, mean={saliency.mean():.4f}")
    
    return elapsed / iterations


def benchmark_grouping():
    """Benchmark the pre-normalized expert grouping."""
    print("\n" + "="*60)
    print("BENCHMARK: Expert Grouping")
    print("="*60)
    
    # Setup test data (reduced size for memory efficiency)
    T, N, D = 512, 32, 2048
    target_ratio = 0.5
    target_experts = int(N * target_ratio)
    
    router_logits = torch.randn(T, N, device='cuda' if torch.cuda.is_available() else 'cpu')
    expert_outputs = torch.randn(N, T, D, device=router_logits.device)
    saliency = torch.rand(N, device=router_logits.device)
    centroid_indices = torch.argsort(saliency, descending=True)[:target_experts]
    
    config = MergeConfig(
        target_ratio=target_ratio,
        group_size=8,
        use_gated_similarity=True,
    )
    
    # Warmup
    for _ in range(3):
        _ = _group_experts_around_centroids(router_logits, expert_outputs, saliency, centroid_indices, config)
    
    # Benchmark
    iterations = 20
    start = time.time()
    for _ in range(iterations):
        groups = _group_experts_around_centroids(router_logits, expert_outputs, saliency, centroid_indices, config)
    elapsed = time.time() - start
    
    print(f"Configuration: T={T}, N={N}, D={D}, target_experts={target_experts}")
    print(f"Time per iteration: {elapsed/iterations*1000:.2f}ms")
    print(f"Number of groups: {len(groups)}")
    print(f"Group sizes: {[len(g) for g in groups[:5]]}...")
    
    return elapsed / iterations


def benchmark_hungarian():
    """Benchmark the Hungarian algorithm permutation solving."""
    print("\n" + "="*60)
    print("BENCHMARK: Hungarian Algorithm Permutation")
    print("="*60)
    
    # Setup test data (reduced size for memory efficiency)
    I, H = 1024, 2048  # intermediate, hidden
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ref = torch.randn(I, 3*H, device=device, dtype=torch.float16 if device.type == 'cuda' else torch.float32)
    candidate = torch.randn(I, 3*H, device=device, dtype=torch.float16 if device.type == 'cuda' else torch.float32)
    
    # Warmup
    for _ in range(3):
        _ = _solve_hungarian_permutation(ref, candidate, device)
    
    # Benchmark
    iterations = 10
    start = time.time()
    for _ in range(iterations):
        perm = _solve_hungarian_permutation(ref, candidate, device)
    elapsed = time.time() - start
    
    print(f"Configuration: I={I}, H={H}, device={device}")
    print(f"Time per iteration: {elapsed/iterations*1000:.2f}ms")
    print(f"Permutation shape: {perm.shape}")
    print(f"Permutation stats: min={perm.min()}, max={perm.max()}")
    
    return elapsed / iterations


def test_correctness():
    """Test that optimizations produce correct results."""
    print("\n" + "="*60)
    print("CORRECTNESS TESTS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1: Saliency computation produces valid scores
    print("\n[Test 1] Saliency computation...")
    T, N, D = 256, 16, 512
    router_logits = torch.randn(T, N, device=device)
    expert_outputs = torch.randn(N, T, D, device=device)
    saliency = _compute_saliency_scores(router_logits, expert_outputs, {}, "saliency_scores", top_k=2)
    assert saliency.shape == (N,), f"Expected shape ({N},), got {saliency.shape}"
    assert saliency.min() >= 0, "Saliency scores should be non-negative"
    print("✓ Saliency computation passed")
    
    # Test 2: Grouping produces valid groups
    print("\n[Test 2] Expert grouping...")
    saliency = torch.rand(N, device=device)
    centroid_indices = torch.argsort(saliency, descending=True)[:4]
    config = MergeConfig(target_ratio=0.25, group_size=4)
    groups = _group_experts_around_centroids(router_logits, expert_outputs, saliency, centroid_indices, config)
    all_experts = set()
    for g in groups:
        all_experts.update(g)
    assert len(all_experts) == N, f"Expected {N} unique experts, got {len(all_experts)}"
    # Only check that non-singleton groups have centroids first
    non_singleton_groups = [g for g in groups if len(g) > 1]
    if non_singleton_groups:
        assert all(g[0] in centroid_indices.tolist() for g in non_singleton_groups), "Centroids should be first in non-singleton groups"
    print("✓ Expert grouping passed")
    
    # Test 3: Hungarian permutation is valid
    print("\n[Test 3] Hungarian permutation...")
    I, H = 64, 128
    ref = torch.randn(I, 3*H, device=device)
    candidate = torch.randn(I, 3*H, device=device)
    perm = _solve_hungarian_permutation(ref, candidate, device)
    assert perm.shape == (I,), f"Expected shape ({I},), got {perm.shape}"
    assert perm.min() == 0 and perm.max() == I-1, f"Permutation should be in range [0, {I-1}]"
    assert len(torch.unique(perm)) == I, "Permutation should have unique indices"
    print("✓ Hungarian permutation passed")
    
    print("\n✓ All correctness tests passed!")


def main():
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("OPTIMIZED MERGE MODULE BENCHMARKS")
    print("="*60)
    
    # Run correctness tests first
    test_correctness()
    
    # Run benchmarks
    benchmark_saliency_computation()
    benchmark_grouping()
    benchmark_hungarian()
    
    print("\n" + "="*60)
    print("BENCHMARKS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
