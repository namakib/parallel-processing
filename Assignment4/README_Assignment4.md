# Assignment 4 — Reductions & Cache-Aware Optimizations (Numba)

**Weight:** 4% | **Individual Work** | **Language:** Python  
**Goal:** Add convergence detection using reductions in Numba, improve cache usage, and optionally apply loop tiling. Compare performance vs fixed-iteration solver from A3.

## 🎯 Learning Outcomes

- Implement parallel reductions in Numba for error computation and convergence detection
- Understand cache-aware optimizations including loop tiling for improved memory locality
- Compare convergence-based vs fixed-iteration approaches for iterative solvers
- Analyze the performance impact of adaptive termination and memory optimizations

## 📋 Requirements & Constraints

✅ **Problem**: Built upon Assignment 3's 2D heat-diffusion with Jacobi iteration  
✅ **Reductions**: Modify Numba kernels to compute updates and accumulate squared error  
✅ **Convergence**: Implement stop when `sqrt(error/M) < 1e-3`  
✅ **Memory**: Ensure contiguous arrays and reuse for optimal cache usage  
✅ **Loop Tiling**: Optional cache-aware tile-based processing  
✅ **Comparison**: Fixed T=500 vs convergence runs with iterations, runtime, speedups  

## 🔧 Implementation Details

### Core Convergence Detection

```python
@njit(parallel=True, fastmath=True, cache=True)
def jacobi_kernel_with_reductions(u: np.ndarray, u_new: np.ndarray) -> float:
    N = u.shape[0]
    total_error = 0.0
    
    for i in prange(1, N-1):
        row_error = 0.0
        for j in range(1, N-1):
            # Compute Jacobi update
            new_val = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
            u_new[i, j] = new_val
            
            # Accumulate squared error
            diff = new_val - u[i, j]
            row_error += diff * diff
        
        total_error += row_error  # Parallel reduction
    
    return total_error
```

### Convergence Algorithm

```python
# Compute RMS error
rms_error = np.sqrt(error_sum / M)  # M = interior points

# Check convergence criterion  
converged = rms_error < threshold  # threshold = 1e-3
```

### Cache-Aware Loop Tiling

```python
@njit(parallel=True, fastmath=True, cache=True)
def jacobi_kernel_with_tiling(u, u_new, tile_size=32) -> float:
    N = u.shape[0]
    total_error = 0.0
        
    for tile_i in prange(1, N-1, tile_size):
        tile_end_i = min(tile_i + tile_size, N-1)
        
        for tile_j in range(1, N-1, tile_size):
            tile_end_j = min(tile_j + tile_size, N-1)
            
            # Process tile interior for cache locality
            for i in range(tile_i, tile_end_i):
                for j in range(tile_j, tile_end_j):
                    # Jacobi update + error accumulation
            total_error += tile_error
    
    return total_error
```

## 🚀 Getting Started

### 1. Installation
```bash
# Install dependencies
pip install -r requirements_a4.txt

# Or individually
pip install numpy matplotlib numba
```

### 2. Test Implementation
```bash
# Verify correct functionality
python test_assignment4.py
```

### 3. Run Assignment
```bash
# Full benchmark (N ∈ {256, 512, 1024})
python Assignment4.py

# Custom parameters
python Assignment4.py --sizes 128 256 512 --threshold 1e-4 --tile-size 64
```

### 4. Results
Output saved to `out_a4/`:
- `runtime.csv` - Performance measurements
- `convergence_analysis.png` - Iterations & runtime vs N  
- `optimization_impact.png` - Detailed optimization analysis
- `report.md` - Comprehensive analysis report
- `report.pdf` - PDF version (if `md-to-pdf` available)

## 📊 Expected Performance Characteristics

### Convergence Detection Benefits
- **Early Termination**: Most problems converge before T=500 iterations
- **Adaptive Performance**: Runtime scales with actual convergence needs
- **Energy Efficiency**: Eliminates unnecessary computation

### Cache Tiling Impact
- **Small Grids (N=256)**: Minimal benefit (already cache-friendly)
- **Medium Grids (N=512)**: Moderate improvement (~1.1-1.3x)
- **Large Grids (N=1024):**: Significant gains (~1.2-1.5x)

### Reduction Performance
- **Parallel Efficiency**: Minimal overhead compared to sequential error computation
- **Memory Access**: Reduction eliminates need for temporary error arrays
- **Scalability**: Benefits scale with number of threads

## 🔍 Technical Comparison

| Optimization | Fixed T=500 | Convergence Detection | + Loop Tiling |
|--------------|-------------|---------------------|---------------|
| **Iterations** | Always 500 | Adaptive (50-400) | Same as convergence |
| **Runtime** | Predictable | Variable by convergence | Further optimized |
| **Memory** | Basic C-contiguous | Same + error reduction | Tile-optimized access |
| **Cache** | Standard | Standard | Enhanced locality |
| **Accuracy**|c-3 Convergence dependent | Threshold-configurable | Same as convergence |

## 📈 Key Insights

### Convergence Detection
1. **Performance Impact**: Varies significantly by problem size and initial conditions
2. **Early Termination**: Provides 2-5x speedup for well-conditioned problems
3. **Robustness**: RMS error provides reliable convergence detection

### Cache Optimization
1. **Loop Tiling**: Improves performance for larger grids where memory bandwidth dominates
2. **Tile Size**: Optimal tile size depends on cache characteristics (test 16, 32, 64)
3. **Memory Layout**: C-contiguous arrays essential for optimal cache performance

### Reduction Efficiency
1. **Parallel Cost**: Minimal overhead compared to sequential error computation
2. **Scalability**: Speedup scales with available CPU cores
3. **Numerical Stability**: Squared error accumulation maintains precision

## 🛠 Advanced Usage

### Custom Convergence Thresholds
```bash
# Tight convergence
python Assignment4.py --threshold 1e-4

# Loose convergence (faster)
python Assignment4.py --threshold 1e-2
```

### Tuning Cache Parameters
```bash
# Large tiles for big grids
python Assignment4.py --tile-size 64 --sizes 1024 2048

# Small tiles for small grids  
python Assignment4.py --tile-size 16 --sizes 128 256
```

### Memory Profiling
```bash
# Test different memory configurations
python -m memory_profiler Assignment4.py
```

## 📝 Deliverables Checklist

- [x] **Source Code**: Convergence detection with reductions
- [x] **Cache Optimization**: Contiguous memory and loop tiling  
- [x] **Performance Comparison**: Fixed vs convergence vs tiled
- [x] **Analysis Plots**: Runtime vs N, iterations to converge
- [x] **Report**: Optimization discussion and impact analysis

## 📚 Evaluation Criteria

- **Correctness (40%)**: Proper convergence detection and cache optimization
- **Optimization (30%)**: Measurable performance improvements from techniques
- **Report Clarity (20%)**: Clear analysis of optimization impact
- **Code Quality (10%)**: Clean implementation with proper error handling

## 🎯 Key Learning Points

1. **Adaptive Algorithms**: Convergence detection can provide dramatic speedups over fixed iterations
2. **Memory Hierarchy**: Cache-aware optimizations become more important for larger problems
3. **Parallel Reductions**: Efficient error computation enables real-time convergence detection
4. **Performance Tuning**: Optimal parameters depend on system architecture and problem characteristics

---

*This assignment demonstrates how modern optimization techniques can transform iterative algorithms from fixed-cost to adaptive and cache-efficient implementations.*
