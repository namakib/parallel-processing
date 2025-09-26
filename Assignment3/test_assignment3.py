#!/usr/bin/env python3
"""
Quick test script for Assignment 3 to verify Numba installation and basic functionality.
"""

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        import numba
        from numba import njit, prange
        print(f"✅ Numba {numba.__version__} imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTo install required packages, run:")
        print("  pip install numpy matplotlib numba")
        print("  # or")
        print("  pip install -r requirements_a3.txt")
        return False

def test_numba_parallel():
    """Test basic Numba parallel functionality"""
    try:
        import numpy as np
        from numba import njit, prange
        
        @njit(parallel=True, fastmath=True)
        def test_kernel(arr):
            n = arr.shape[0]
            for i in prange(n):
                arr[i] = i * 2.0
        
        # Test on small array
        test_arr = np.zeros(100, dtype=np.float64)
        test_kernel(test_arr)
        
        expected = np.arange(100) * 2.0
        if np.allclose(test_arr, expected):
            print("✅ Numba parallel execution test passed")
            return True
        else:
            print("❌ Numba parallel execution test failed")
            return False
            
    except Exception as e:
        print(f"❌ Numba test error: {e}")
        return False

def main():
    print("=== Assignment 3: Numba Setup Test ===\n")
    
    if not test_imports():
        return 1
        
    if not test_numba_parallel():
        return 1
        
    print("\n✅ All tests passed! Assignment 3 is ready to run.")
    print("\nTo run Assignment 3:")
    print("  python Assignment3.py")
    print("  python Assignment3.py --sizes 256 512 --iters 100  # smaller test")
    
    return 0

if __name__ == "__main__":
    exit(main())
