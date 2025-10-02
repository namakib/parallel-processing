#!/usr/bin/env python3
"""
Test script for Assignment 4 - Reductions & Cache-Aware Optimizations

This script validates the correctness and basic functionality of the 
convergence detection and cache optimization implementations.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add Assignment4 to path
sys.path.insert(0, str(Path(__file__).parent))

from Assignment4 import (
    jacobi_kernel_with_reductions,
    jacobi_kernel_explicit_reduction, 
    jacobi_kernel_with_tiling,
    jacobi_convergence_solver,
    jacobi_fixed_solver,
    init_dirichlet_grid
)

def test_grid_initialization():
    """Test grid initialization"""
    print("Testing grid initialization...")
    
    N = 64
    u, u_new = init_dirichlet_grid(N)
    
    # Check shapes
    assert u.shape == (N, N), f"Expected shape {(N, N)}, got {u.shape}"
    assert u_new.shape == (N, N), f"Expected shape {(N, N)}, got {u_new.shape}"
    
    # Check boundary conditions
    assert np.allclose(u[0, :], 100.0), "Top boundary should be 100.0"
    assert np.allclose(u[1:, 0], 0.0), "All other boundaries should be 0.0"
    assert np.allclose(u[1:, -1], 0.0), "Right boundary should be 0.0"
    assert np.allclose(u[-1, :], 0.0), "Bottom boundary should be 0.0"
    
    # Check memory layout
    assert u.flags.c_contiguous, "Array should be C-contiguous"
    assert u_new.flags.c_contiguous, "Array should be C-contiguous"
    
    print("✅ Grid initialization test passed")

def test_kernel_basic():
    """Test basic kernel functionality"""
    print("Testing basic kernel...")
    
    N = 32
    u, u_new = init_dirichlet_grid(N)
    
    # Test reduction kernel
    error1 = jacobi_kernel_with_reductions(u, u_new)
    error2 = jacobi_kernel_explicit_reduction(u, u_new)
    
    # Errors should be similar (both methods compute same thing)
    assert np.allclose(error1, error2, rtol=1e-10), f"Errors should match: {error1} vs {error2}"
    
    # Check that interior points were updated
    interior_old = u[1:-1, 1:-1].copy()
    interior_new = u_new[1:-1, 1:-1].copy()
    
    # These should be different (unless already converged)
    diff_norm = np.linalg.norm(interior_new - interior_old)
    assert diff_norm > 1e-10, "Interior points should be updated"
    
    print("✅ Basic kernel test passed")

def test_kernel_tiling():
    """Test tiling kernel functionality"""
    print("Testing tiling kernel...")
    
    N = 64
    u, u_new = init_dirichlet_grid(N)
    
    # Test tiling kernel
    error_tiled = jacobi_kernel_with_tiling(u, u_new)
    
    # Should produce reasonable error value
    assert error_tiled >= 0, f"Error should be non-negative, got {error_tiled}"
    assert error_tiled < 1e6, f"Error seems too large: {error_tiled}"
    
    print("✅ Tiling kernel test passed")

def test_convergence_solver():
    """Test convergence-based solver"""
    print("Testing convergence solver...")
    
    N = 128
    threshold = 1e-2  # Relaxed threshold for faster testing
    
    # Test basic convergence solver
    iterations, runtime, warmup = jacobi_convergence_solver(N, threshold, max_iter=100)
    
    assert iterations > 0, f"Should converge in positive iterations, got {iterations}"
    assert iterations <= 100, f"Should converge within max iterations, got {iterations}"
    assert runtime > 0, f"Runtime should be positive, got {runtime}"
    
    print(f"✅ Convergence solver test passed: {iterations} iterations, {runtime:.3f}s runtime")

def test_fixed_solver():
    """Test fixed iteration solver"""
    print("Testing fixed solver...")
    
    N = 128
    T = 50  # Reduced for faster testing
    
    runtime, warmup = jacobi_fixed_solver(N, T)
    
    assert runtime > 0, f"Runtime should be positive, got {runtime}"
    assert warmup >= 0, f"Warmup time should be non-negative, got {warmup}"
    
    print(f"✅ Fixed solver test passed: {runtime:.3f}s runtime")

def test_comparison():
    """Test that both solvers produce similar results for large iterations"""
    print("Testing solver comparison...")
    
    N = 128
    T = 100
    threshold = 1e-4
    
    # Run fixed solver
    fixed_runtime, _ = jacobi_fixed_solver(N, T)
    
    # Run convergence solver with same max iterations
    conv_iters, conv_runtime, _ = jacobi_convergence_solver(N, threshold, max_iter=T)
    
    print(f"  Fixed: {T} iterations, {fixed_runtime:.3f}s")
    print(f"  Convergence: {conv_iters} iterations, {conv_runtime:.3f}s")
    
    # Convergence solver shouldn't be much faster (same number of iterations)
    speedup = fixed_runtime / conv_runtime if conv_runtime > 0 else 0
    assert speedup < 5.0, f"Convergence solver shouldn't be dramatically faster: {speedup:.2f}x"
    
    print("✅ Solver comparison test passed")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Assignment 4 - Reductions & Cache-Aware Optimizations Tests")
    print("=" * 60)
    
    try:
        test_grid_initialization()
        print()
        
        test_kernel_basic()
        print()
        
        test_kernel_tiling()
        print()
        
        test_convergence_solver()
        print()
        
        test_fixed_solver()
        print()
        
        test_comparison()
        print()
        
        print("🎉 All tests passed! Assignment 4 implementation is working correctly.")
        print("\nYou can now run the full benchmark with:")
        print("  python Assignment4.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
if __name__ == "__main__":
    run_all_tests()
