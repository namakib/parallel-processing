# Parallel Processing Assignments

This repository contains three assignments exploring different approaches to parallelizing a 2D heat diffusion solver using the Jacobi iteration method.

## 📁 Project Structure

```
ParallelProcessing/
├── Assignment1/
│   └── Assignment1.py          # Sequential baseline implementation
├── Assignment2/
│   └── Assignment2.py          # Multiprocessing with shared memory
├── Assignment3/
│   ├── Assignment3.py          # Numba JIT parallel implementation
│   ├── README_Assignment3.md   # Detailed Assignment 3 documentation
│   ├── requirements_a3.txt     # Python dependencies for Assignment 3
│   └── test_assignment3.py     # Setup verification script
├── venv/                       # Virtual environment (all dependencies)
├── activate_env.sh             # Environment activation script
├── pyrightconfig.json          # Type checker configuration (basedpyright/pyright)
├── pyproject.toml              # Project metadata and tool configuration
├── markdown.css                # Styling for generated reports
└── README.md                   # This overview file
```

## 🎯 Assignment Overview

### Assignment 1: Sequential Baseline
- **Location**: `Assignment1/Assignment1.py`
- **Approach**: Pure Python/NumPy sequential implementation
- **Purpose**: Establishes performance baseline for comparison

### Assignment 2: Process-Based Parallelism  
- **Location**: `Assignment2/Assignment2.py`
- **Approach**: `multiprocessing` with shared memory and explicit halo exchange
- **Features**: Process pool, shared memory arrays, barrier synchronization
- **Weight**: 4% | Individual Work

### Assignment 3: Numba JIT Parallelism
- **Location**: `Assignment3/Assignment3.py` 
- **Approach**: Numba `@njit(parallel=True)` with OpenMP-style `prange` loops
- **Features**: JIT compilation, thread-based parallelism, LLVM optimization
- **Weight**: 4% | Individual Work

## 🚀 Quick Start

### Setup Virtual Environment (Recommended)
```bash
# Virtual environment is already set up with all dependencies
source activate_env.sh        # Activate environment
# or manually: source venv/bin/activate

# To deactivate when done
deactivate
```

### Type Checking Configuration
The project includes configuration for [basedpyright](https://docs.basedpyright.com/v1.31.3/configuration/config-files/#reportMissingImports) (an enhanced Python type checker):
- **`pyrightconfig.json`**: JSON configuration with virtual environment path
- **`pyproject.toml`**: Modern Python project configuration with type checking rules
- **Import resolution**: Properly configured to find packages in the virtual environment
- **Relaxed rules**: Tuned for scientific computing workflows (allows untyped NumPy operations)

### Manual Installation (Alternative)
```bash
# For all assignments
pip install numpy matplotlib

# Additional for Assignment 3  
pip install numba
```

### Running Assignments

#### Assignment 1 (Sequential)
```bash
cd Assignment1
python Assignment1.py
```

#### Assignment 2 (Multiprocessing)
```bash
cd Assignment2  
python Assignment2.py
```

#### Assignment 3 (Numba Parallel)
```bash
cd Assignment3

# Test setup first
python test_assignment3.py

# Run assignment  
python Assignment3.py

# Quick test with smaller parameters
python Assignment3.py --sizes 256 512 --iters 100
```

## 📊 Performance Comparison

Each assignment solves the same problem with different parallelization strategies:

| Aspect | Assignment 1 | Assignment 2 | Assignment 3 |
|--------|-------------|-------------|-------------|
| **Approach** | Sequential | Multiprocessing | Numba JIT |
| **Parallelism** | None | Process-based | Thread-based |
| **Memory** | Local arrays | Shared memory | Shared threads |
| **Overhead** | Minimal | Process sync | JIT compilation |
| **Scaling** | N/A | Limited by communication | Limited by threads |

## 🔧 Common Parameters

All assignments use consistent parameters for fair comparison:
- **Grid sizes**: N ∈ {256, 512, 1024}
- **Iterations**: T = 500  
- **Problem**: 2D heat diffusion with Dirichlet boundaries
- **Algorithm**: Jacobi iteration with 4-point stencil

## 📈 Expected Performance Trends

- **Assignment 1**: Baseline performance, scales with O(N²) per iteration
- **Assignment 2**: May show speedup for large N, but overhead dominates for small N
- **Assignment 3**: Best speedup potential, especially for CPU-bound workloads

## 📝 Output and Reports

Each assignment generates:
- Performance timing data (CSV)
- Runtime vs problem size plots (PNG)
- Detailed analysis reports (Markdown/PDF)
- Console summaries

## 🛠 Development Notes

### Problem Setup
- **Boundaries**: Top=100°, others=0° (Dirichlet conditions)
- **Interior**: Updated using 5-point Jacobi stencil
- **Convergence**: Fixed 500 iterations (no tolerance checking)

### Key Implementation Details
- All implementations use double precision (float64)
- Array swapping avoids expensive copies
- Boundary conditions reapplied each iteration
- Timing excludes initialization and visualization

## 📚 Learning Objectives

1. **Sequential optimization**: Understand NumPy vectorization and memory access patterns
2. **Process parallelism**: Learn shared memory, synchronization, and communication overhead
3. **JIT compilation**: Experience modern Python acceleration with Numba
4. **Performance analysis**: Compare different parallelization strategies systematically
5. **Scientific computing**: Apply parallel techniques to real numerical algorithms

---

*These assignments demonstrate the evolution from sequential to parallel computing approaches, highlighting the trade-offs between implementation complexity, performance gains, and development productivity.*
