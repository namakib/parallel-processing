# Assignment 3 — OpenMP-Style Parallel Loops with Numba

**Weight:** 4% | **Individual Work** | **Language:** Python  
**Goal:** Convert sequential 2D heat-diffusion (Jacobi) solver into a compiled, parallel version using Numba's `@njit(parallel=True)` with `prange`.

## 🎯 Learning Outcomes

- Achieve OpenMP-like loop parallelism from Python using Numba JIT compilation
- Understand the performance benefits and trade-offs of JIT compilation
- Compare different parallelization approaches (sequential, multiprocessing, Numba threads)
- Measure and analyze parallel speedups with proper benchmarking methodology

## 📋 Requirements & Constraints

✅ **Same problem**: 2D heat-diffusion with Dirichlet boundaries  
✅ **Algorithm**: Jacobi iteration with two arrays, swap each iteration  
✅ **Fixed iterations**: T=500  
✅ **Numba directives**: `@njit(parallel=True, fastmath=True)`  
✅ **JIT warm-up**: Exclude compilation time from measurements  

## 🔧 Implementation Details

### Core Numba Kernel
```python
@njit(parallel=True, fastmath=True, cache=True)
def jacobi_kernel(u: np.ndarray, u_new: np.ndarray) -> None:
    N = u.shape[0]
    # Parallel loop over interior rows
    for i in prange(1, N-1):
        for j in range(1, N-1):
            u_new[i, j] = 0.25 * (
                u[i-1, j] +     # up
                u[i+1, j] +     # down  
                u[i, j-1] +     # left
                u[i, j+1]       # right
            )
```

### Key Optimizations
1. **C-contiguous arrays**: `order='C'` for optimal memory layout
2. **Pre-allocation**: Arrays reused across all iterations
3. **JIT warm-up**: 5 iterations to compile and optimize before timing
4. **Fast math**: Aggressive floating-point optimizations
5. **Function caching**: Compiled functions cached for subsequent runs

## 🚀 Getting Started

### 1. Install Dependencies
```bash
# Install required packages
pip install numpy matplotlib numba

# Or using requirements file
pip install -r requirements_a3.txt
```

### 2. Test Setup
```bash
# Verify Numba installation and parallel execution
python test_assignment3.py
```

### 3. Run Assignment
```bash
# Full benchmark (N ∈ {256, 512, 1024}, T=500)
python Assignment3.py

# Quick test (smaller parameters)
python Assignment3.py --sizes 256 512 --iters 100

# Custom parameters
python Assignment3.py --sizes 256 512 1024 --iters 500 --outdir results_a3
```

### 4. View Results
Results are saved to `out_a3/` (or specified `--outdir`):
- `runtime.csv` - Raw performance data
- `runtime_comparison.png` - Sequential vs Numba performance
- `speedup_analysis.png` - Detailed speedup analysis
- `report.md` - Comprehensive analysis report
- `report.pdf` - PDF report (if `md-to-pdf` available)

## 📊 Expected Performance

### Typical Speedups (will vary by system)
- **N=256**: ~2-4× speedup over sequential
- **N=512**: ~4-6× speedup over sequential  
- **N=1024**: ~6-10× speedup over sequential

### Performance Factors
- **CPU cores**: More cores → better parallel performance
- **Memory bandwidth**: Large grids become memory-bound
- **Cache effects**: C-contiguous arrays optimize cache usage
- **JIT compilation**: One-time cost, amortized over iterations

## 🔍 Comparison with Previous Assignments

| Aspect | A1: Sequential | A2: Multiprocessing | A3: Numba |
|--------|---------------|-------------------|-----------|
| **Parallelism** | None | Process-based | Thread-based |
| **Memory** | Local arrays | Shared memory | Shared threads |
| **Communication** | N/A | Explicit halo exchange | Implicit |
| **Compilation** | Interpreted | Interpreted | JIT compiled |
| **Overhead** | Minimal | Process startup/sync | JIT compilation |
| **Scaling** | N/A | Limited by sync/comm | Limited by threads/memory |

## 📝 Deliverables

1. **Source Code**: `Assignment3.py` - Numba-parallel solver
2. **Performance Plots**: Runtime comparison and speedup analysis
3. **Report**: Setup description, speedup analysis, JIT cost discussion
4. **Data**: CSV file with all timing measurements

## 📈 Evaluation Criteria

- **Correctness (40%)**: Proper implementation of Numba parallel kernel
- **Parallel Speedup (30%)**: Measurable performance improvement over sequential
- **Report Clarity (20%)**: Clear analysis of results and trade-offs
- **Code Quality (10%)**: Clean, well-documented, efficient implementation

## 🛠 Technical Notes

### Numba Compilation Process
1. **First call**: Python bytecode → LLVM IR → machine code
2. **Warm-up runs**: Optimization passes, type inference
3. **Subsequent calls**: Direct execution of compiled code

### Memory Layout Optimization
```python
# C-contiguous arrays for optimal cache performance
u = np.zeros((N, N), dtype=np.float64, order='C')
u_new = np.zeros_like(u, order='C')
```

### Parallel Loop Distribution
- `prange(1, N-1)` distributes loop iterations across available threads
- Each thread processes a contiguous block of rows
- No explicit synchronization needed within the kernel
- Boundaries handled separately to avoid race conditions

## 🎯 Learning Tips

1. **Understand JIT cost**: First run is slow (compilation), subsequent runs are fast
2. **Profile different N**: See how speedup scales with problem size  
3. **Monitor CPU usage**: Verify all cores are utilized during computation
4. **Compare memory patterns**: C-contiguous vs Fortran-contiguous performance
5. **Experiment with fastmath**: Observe precision vs performance trade-offs

## 📚 Additional Resources

- [Numba Documentation](https://numba.readthedocs.io/)
- [Numba Parallel Computing](https://numba.readthedocs.io/en/stable/user/parallel.html)
- [LLVM and JIT Compilation](https://llvm.org/docs/tutorial/)

---

*This assignment demonstrates how modern Python tools like Numba can bridge the performance gap between interpreted and compiled languages while maintaining code simplicity and readability.*
