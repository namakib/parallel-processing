from __future__ import annotations
import argparse, os, csv, time, shutil, subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Numba imports
from numba import njit, prange

# Multiprocessing imports (for A2 comparison)
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

# =============================
# Config & CLI
# =============================
@dataclass
class RunConfig:
    sizes: Tuple[int, ...]           # N ∈ {256,512,1024}
    iters: int                       # T = 500 per assignment
    outdir: str                      # output directory
    make_pdf: bool = True            # convert MD -> PDF via md-to-pdf CLI
    pdf_timeout: int = 30            # seconds
    warmup_iters: int = 5            # JIT warm-up iterations (not timed)

# =============================
# Core: Dirichlet grid init
# Top boundary = 100, others = 0
# =============================
def init_dirichlet_grid(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize C-contiguous arrays for Dirichlet boundary conditions"""
    # Ensure C-contiguous arrays
    u = np.zeros((N, N), dtype=np.float64, order='C')
    u_new = np.zeros_like(u, order='C')
    u[0, :] = 100.0
    u_new[0, :] = 100.0
    return u, u_new

# =============================
# Numba JIT kernel for Jacobi iteration
# =============================
@njit(parallel=True, fastmath=True, cache=True)
def jacobi_kernel(u: np.ndarray, u_new: np.ndarray) -> None:
    """
    Numba JIT-compiled Jacobi kernel with parallel loops.
    Updates interior cells using 4-point stencil with prange for parallelization.
    
    Args:
        u: Current grid (input)
        u_new: Next grid (output)
    """
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

@njit(parallel=False, fastmath=True, cache=True)
def apply_boundaries(u: np.ndarray) -> None:
    """Apply Dirichlet boundary conditions (boundaries are fixed)"""
    N = u.shape[0]
    u[0, :] = 100.0    # top = 100
    u[-1, :] = 0.0     # bottom = 0
    u[:, 0] = 0.0      # left = 0  
    u[:, -1] = 0.0     # right = 0

# =============================
# A2 Multiprocessing Implementation (for comparison)
# =============================
def partition_rows(N: int, W: int) -> List[Tuple[int, int]]:
    """Partition interior rows for multiprocessing"""
    start = 1
    end = N - 1
    total = end - start
    base = total // W
    rem = total % W
    parts = []
    cur = start
    for i in range(W):
        add = base + (1 if i < rem else 0)
        parts.append((cur, cur + add))
        cur += add
    return parts

# Worker globals for A2 multiprocessing
_g_shape: Tuple[int, int] | None = None
_g_u: np.ndarray | None = None
_g_un: np.ndarray | None = None
_g_chunks: Dict[int, np.ndarray] = {}
_g_main_shms: List[shared_memory.SharedMemory] = []
_g_chunk_shms: List[shared_memory.SharedMemory] = []

def worker_initializer(u_name: str, un_name: str, shape: Tuple[int, int], 
                      chunk_names: List[str], chunk_shapes: List[Tuple[int, int]]):
    """Initialize worker with shared memory arrays"""
    global _g_shape, _g_u, _g_un, _g_chunks, _g_main_shms, _g_chunk_shms
    _g_shape = shape
    
    shm_u = shared_memory.SharedMemory(name=u_name)
    shm_un = shared_memory.SharedMemory(name=un_name)
    _g_main_shms = [shm_u, shm_un]
    _g_u = np.ndarray(shape, dtype=np.float64, buffer=shm_u.buf)
    _g_un = np.ndarray(shape, dtype=np.float64, buffer=shm_un.buf)
    
    _g_chunks = {}
    _g_chunk_shms = []
    for i, nm in enumerate(chunk_names):
        shm = shared_memory.SharedMemory(name=nm)
        _g_chunk_shms.append(shm)
        _g_chunks[i] = np.ndarray(chunk_shapes[i], dtype=np.float64, buffer=shm.buf)

def jacobi_chunk_with_halo(args: Tuple[int, int, int]):
    """Worker function for multiprocessing Jacobi iteration"""
    global _g_un, _g_chunks, _g_shape
    idx, r0, r1 = args
    halo = _g_chunks[idx]
    N, _ = _g_shape
    
    _g_un[r0:r1, 1:N-1] = 0.25 * (
        halo[0:-2, 1:N-1] +  # up
        halo[2:, 1:N-1] +    # down
        halo[1:-1, 0:N-2] +  # left
        halo[1:-1, 2:]       # right
    )

def jacobi_multiprocessing_run(N: int, T: int, W: int = 4) -> float:
    """A2 multiprocessing implementation for comparison"""
    u, u_new = init_dirichlet_grid(N)
    
    shm_u = shared_memory.SharedMemory(create=True, size=u.nbytes)
    shm_un = shared_memory.SharedMemory(create=True, size=u_new.nbytes)
    u_sh = np.ndarray(u.shape, dtype=u.dtype, buffer=shm_u.buf)
    un_sh = np.ndarray(u_new.shape, dtype=u_new.dtype, buffer=shm_un.buf)
    u_sh[:] = u
    un_sh[:] = u_new
    
    parts = partition_rows(N, W)
    chunk_shms = []
    chunk_shapes = []
    for (r0, r1) in parts:
        H = (r1 - r0) + 2
        shm = shared_memory.SharedMemory(create=True, size=(H * N * np.float64().nbytes))
        chunk_shms.append(shm)
        chunk_shapes.append((H, N))
        np.ndarray((H, N), dtype=np.float64, buffer=shm.buf).fill(0.0)
    
    chunk_names = [shm.name for shm in chunk_shms]
    
    try:
        with ProcessPoolExecutor(max_workers=W, initializer=worker_initializer,  # type: ignore
                                initargs=(shm_u.name, shm_un.name, (u.shape[0], u.shape[1]), chunk_names, chunk_shapes)) as pool:
            
            def fill_halos_from_u():
                for i, (r0, r1) in enumerate(parts):
                    halo = np.ndarray(chunk_shapes[i], dtype=np.float64, buffer=chunk_shms[i].buf)
                    halo[1:-1, :] = u_sh[r0:r1, :]
                    top_src = r0 - 1 if r0 - 1 >= 0 else 0
                    halo[0, :] = u_sh[top_src, :]
                    bottom_src = r1 if r1 < N else N - 1
                    halo[-1, :] = u_sh[bottom_src, :]
            
            t0 = time.perf_counter()
            for _ in range(T):
                fill_halos_from_u()
                futures = [pool.submit(jacobi_chunk_with_halo, (i, r0, r1)) 
                          for i, (r0, r1) in enumerate(parts)]
                wait(futures, return_when=ALL_COMPLETED)
                
                un_sh[0, :] = 100.0; un_sh[-1, :] = 0.0; un_sh[:, 0] = 0.0; un_sh[:, -1] = 0.0
                u_sh[:], un_sh[:] = un_sh, u_sh
            t1 = time.perf_counter()
    
    finally:
        shm_u.close(); shm_un.close()
        shm_u.unlink(); shm_un.unlink()
        for shm in chunk_shms:
            try:
                shm.close(); shm.unlink()
            except Exception:
                pass
    
    return t1 - t0

# =============================
# Sequential baseline (for comparison)
# =============================
def jacobi_sequential_run(N: int, T: int) -> float:
    """Sequential implementation without Numba for baseline comparison"""
    u, u_new = init_dirichlet_grid(N)
    
    t0 = time.perf_counter()
    for _ in range(T):
        u_new[1:-1, 1:-1] = 0.25 * (
            u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
        )
        # Apply fixed boundaries
        u_new[0, :] = 100.0
        u_new[-1, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0
        u, u_new = u_new, u
    t1 = time.perf_counter()
    return t1 - t0

# =============================
# Numba parallel run
# =============================
def jacobi_numba_run(N: int, T: int, warmup: int = 5) -> Tuple[float, float]:
    """
    Numba-accelerated parallel Jacobi solver.
    
    Returns:
        Tuple of (runtime_seconds, warmup_time_seconds)
    """
    # Pre-allocate C-contiguous arrays
    u, u_new = init_dirichlet_grid(N)
    
    # JIT warm-up (compile and optimize)
    print(f"  Warming up JIT for N={N}...")
    warmup_start = time.perf_counter()
    for _ in range(warmup):
        jacobi_kernel(u, u_new)
        apply_boundaries(u_new) 
        u, u_new = u_new, u
    warmup_time = time.perf_counter() - warmup_start
    
    # Timed computation (exclude warm-up, I/O, plotting)
    print(f"  Running {T} timed iterations...")
    t0 = time.perf_counter()
    for _ in range(T):
        jacobi_kernel(u, u_new)
        apply_boundaries(u_new)
        u, u_new = u_new, u
    t1 = time.perf_counter()
    
    return t1 - t0, warmup_time

# =============================
# Helpers: CSV / plotting / report 
# =============================
def ensure_outdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    ensure_outdir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def plot_runtime_comparison(rows: List[Dict], path: str) -> None:
    """Plot runtime comparison: A1 (sequential) vs A2 (multiprocessing) vs A3 (Numba parallel)"""
    ensure_outdir(os.path.dirname(path))
    plt.figure(figsize=(12, 6))
    
    Ns = sorted({r["N"] for r in rows})
    
    # Collect results for all three approaches
    seq_times = []
    mp_times = []
    numba_times = []
    
    for N in Ns:
        seq_row = next((r for r in rows if r["N"] == N and r["type"] == "sequential"), None)
        mp_row = next((r for r in rows if r["N"] == N and r["type"] == "multiprocessing"), None)
        numba_row = next((r for r in rows if r["N"] == N and r["type"] == "numba_parallel"), None)
        
        seq_times.append(seq_row["time_sec"] if seq_row else 0)
        mp_times.append(mp_row["time_sec"] if mp_row else 0)
        numba_times.append(numba_row["time_sec"] if numba_row else 0)
    
    # Bar plot comparison
    x = np.arange(len(Ns))
    width = 0.25
    
    plt.bar(x - width, seq_times, width, label='A1: Sequential', alpha=0.8, color='blue')
    plt.bar(x, mp_times, width, label='A2: Multiprocessing', alpha=0.8, color='orange')
    plt.bar(x + width, numba_times, width, label='A3: Numba Parallel', alpha=0.8, color='green')
    
    plt.xlabel('Grid Size (N)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Jacobi Heat Diffusion: A1 vs A2 vs A3 Runtime Comparison (T=500)')
    plt.xticks(x, [f'{N}' for N in Ns])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_speedup_analysis(rows: List[Dict], path: str) -> None:
    """Plot speedup and efficiency analysis for all three approaches"""
    ensure_outdir(os.path.dirname(path))
    plt.figure(figsize=(15, 5))
    
    Ns = sorted({r["N"] for r in rows})
    
    # Subplot 1: Runtime comparison
    plt.subplot(1, 3, 1)
    for type_name, label, style, color in [
        ("sequential", "A1: Sequential", "o-", "blue"),
        ("multiprocessing", "A2: Multiprocessing", "s--", "orange"),
        ("numba_parallel", "A3: Numba Parallel", "^:", "green")
    ]:
        type_rows = [r for r in rows if r["type"] == type_name]
        type_rows = sorted(type_rows, key=lambda x: x["N"])
        xs = [r["N"] for r in type_rows]
        ys = [r["time_sec"] for r in type_rows]
        if xs and ys:
            plt.plot(xs, ys, style, label=label, linewidth=2, markersize=6, color=color)
    
    plt.xlabel('Grid Size (N)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Grid Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Subplot 2: A2 Speedup vs A1
    plt.subplot(1, 3, 2)
    mp_speedups = []
    for N in Ns:
        seq_row = next((r for r in rows if r["N"] == N and r["type"] == "sequential"), None)
        mp_row = next((r for r in rows if r["N"] == N and r["type"] == "multiprocessing"), None)
        
        if seq_row and mp_row:
            speedup = seq_row["time_sec"] / mp_row["time_sec"]
            mp_speedups.append(speedup)
        else:
            mp_speedups.append(0)
    
    plt.bar(range(len(Ns)), mp_speedups, alpha=0.7, color='orange')
    plt.xlabel('Grid Size')
    plt.ylabel('Speedup (A1/A2)')
    plt.title('Multiprocessing Speedup')
    plt.xticks(range(len(Ns)), [f'N={N}' for N in Ns])
    plt.grid(True, alpha=0.3)
    
    # Add speedup values on bars
    for i, v in enumerate(mp_speedups):
        if v > 0:
            plt.text(i, v + 0.05, f'{v:.1f}x', ha='center', va='bottom')
    
    # Subplot 3: A3 Speedup vs A1
    plt.subplot(1, 3, 3)
    numba_speedups = []
    for N in Ns:
        seq_row = next((r for r in rows if r["N"] == N and r["type"] == "sequential"), None)
        numba_row = next((r for r in rows if r["N"] == N and r["type"] == "numba_parallel"), None)
        
        if seq_row and numba_row:
            speedup = seq_row["time_sec"] / numba_row["time_sec"]
            numba_speedups.append(speedup)
        else:
            numba_speedups.append(0)
    
    plt.bar(range(len(Ns)), numba_speedups, alpha=0.7, color='green')
    plt.xlabel('Grid Size')
    plt.ylabel('Speedup (A1/A3)')
    plt.title('Numba Parallel Speedup')
    plt.xticks(range(len(Ns)), [f'N={N}' for N in Ns])
    plt.grid(True, alpha=0.3)
    
    # Add speedup values on bars
    for i, v in enumerate(numba_speedups):
        if v > 0:
            plt.text(i, v + 0.05, f'{v:.1f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def build_interpretation(runtime_rows: List[Dict]) -> str:
    """Generate interpretation of Numba parallel results vs sequential"""
    seq_rows = [r for r in runtime_rows if r["type"] == "sequential"]
    numba_rows = [r for r in runtime_rows if r["type"] == "numba_parallel"]
    
    lines: List[str] = []
    
    # Performance analysis by grid size
    Ns = sorted({r["N"] for r in runtime_rows})
    best_speedup = 0.0
    best_N = None
    
    for N in Ns:
        seq = next((r for r in seq_rows if r["N"] == N), None)
        numba = next((r for r in numba_rows if r["N"] == N), None)
        
        if seq and numba:
            speedup = seq["time_sec"] / numba["time_sec"]
            if speedup > best_speedup:
                best_speedup = speedup
                best_N = N
            
            lines.append(
                f"- **N={N}**: Sequential = {seq['time_sec']:.3f}s, "
                f"Numba = {numba['time_sec']:.3f}s → **{speedup:.1f}× speedup**"
            )
    
    # Overall summary
    if best_N:
        lines.insert(0, 
            f"**Best speedup achieved**: **{best_speedup:.1f}×** at **N={best_N}**\n"
        )
    
    lines.extend([
        "",
        "### Analysis:",
        "- **JIT Compilation**: Numba compiles Python to optimized machine code with LLVM backend.",
        "- **Parallel Execution**: `prange` enables OpenMP-like parallelization across CPU cores.",
        "- **Memory Layout**: C-contiguous arrays optimize cache performance and SIMD vectorization.",  
        "- **Fastmath**: Aggressive floating-point optimizations improve computational throughput.",
        "",
        "### Scaling Characteristics:",
        "- **Compute-bound**: Larger grids show better speedup as parallelism overhead is amortized.",
        "- **Memory bandwidth**: Performance ultimately limited by memory access patterns.",
        "- **Thread overhead**: Smaller grids may show diminishing returns due to thread management costs."
    ])
    
    return "\n".join(lines)

def write_report_md(runtime_rows: List[Dict], iters: int, outdir: str, make_pdf: bool, pdf_timeout: int):
    os.makedirs(outdir, exist_ok=True)
    md_path = os.path.join(outdir, "report.md")
    
    # Build results table
    headers = ["N", "Type", "Runtime (sec)", "Speedup vs A1", "JIT Warmup (sec)"]
    lines_table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    
    Ns = sorted({r["N"] for r in runtime_rows})
    for N in Ns:
        N_rows = sorted([r for r in runtime_rows if r["N"] == N], 
                        key=lambda x: (x["type"] == "numba_parallel", x["type"]))
        for r in N_rows:
            speedup_str = f"{r.get('speedup_vs_seq', 1.0):.2f}×" if 'speedup_vs_seq' in r else "1.00×"
            warmup_str = f"{r.get('warmup_time', 0.0):.3f}" if 'warmup_time' in r else "N/A"
            type_display = "Sequential" if r["type"] == "sequential" else "Numba Parallel"
            lines_table.append(
                f"| {r['N']} | {type_display} | {r['time_sec']:.3f} | {speedup_str} | {warmup_str} |"
            )
    
    interpretation = build_interpretation(runtime_rows)
    
    md_content = [
        "# Assignment 3 – OpenMP-Style Parallel Loops with Numba",
        "",
        f"**2D Heat Diffusion Solver: A1 vs A2 vs A3 Comparison (T = {iters})**",
        "",
        "## Setup",
        "",
        "**Dependencies**: `numpy>=1.21.0`, `matplotlib>=3.5.0`, `numba>=0.56.0`",
        "",
        "**Installation**: `pip install numpy matplotlib numba`",
        "",
        "**Run**: `python Assignment3.py` (generates results in `out_a3/` directory)",
        "",
        "## Assignment Comparison",
        "- **A1 (Sequential)**: Pure Python/NumPy implementation",
        "- **A2 (Multiprocessing)**: Process-based parallelism with shared memory",
        "- **A3 (Numba Parallel)**: JIT compilation with OpenMP-style `prange` loops",
        "",
        "## Performance Results",
        "",
        "### Runtime Comparison",
        "![Runtime Analysis](speedup_analysis.png)",
        "",
        "### Results Table",
        *lines_table,
        "",
        "## JIT Compilation Cost",
        "- **Warm-up Phase**: 5 iterations to trigger JIT compilation and optimization",
        "- **One-time Cost**: JIT compilation happens once per function signature",
        "- **Measurement**: Warm-up time excluded from performance benchmarks",
        "- **Production**: In real applications, JIT cost is amortized over many calls",
        "",
        "## Comparison with Previous Assignments",
        "- **A1 (Sequential)**: Pure Python numpy operations",
        "- **A2 (Multiprocessing)**: Process-based parallelism with shared memory",  
        "- **A3 (Numba)**: Thread-based parallelism with JIT compilation",
        "",
        "### Trade-offs",
        "- **Compilation overhead**: JIT warm-up vs immediate execution",
        "- **Memory sharing**: Threads vs processes",
        "- **Scaling**: Thread synchronization vs process communication",
        "",
        "## Conclusion",
        "Numba provides an excellent balance of performance and simplicity for computational kernels. "
        "The `@njit(parallel=True)` decorator enables OpenMP-like parallelization with minimal code changes, "
        "while LLVM compilation delivers near-C performance from Python. For iterative algorithms like Jacobi, "
        "the JIT compilation cost is easily amortized, making Numba an attractive option for scientific computing.",
    ]
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"✅ Wrote {md_path}")
    
    # Convert to PDF via CLI `md-to-pdf` if available
    cli = shutil.which("md-to-pdf")
    if make_pdf and cli is not None:
        try:
            # Use markdown.css from project root (one level up from Assignment3)
            css_path = os.path.join("..", "markdown.css")
            subprocess.run(
                ["md-to-pdf", "report.md", "--stylesheet", css_path],
                cwd=outdir,
                check=True,
                timeout=pdf_timeout,
            )
            print(f"✅ PDF created at {os.path.join(outdir,'report.pdf')} (with custom CSS)")
        except subprocess.TimeoutExpired:
            print(f"⏱️ md-to-pdf timed out after {pdf_timeout}s — skipping PDF")
        except subprocess.CalledProcessError as e:
            print("⚠️ md-to-pdf failed:", e)
    elif make_pdf:
        print("⚠️ Skipping PDF: `md-to-pdf` not found. Install with: npm install -g md-to-pdf")

# =============================
# Benchmarks
# =============================
def run_benchmarks(cfg: RunConfig) -> List[Dict]:
    """Run benchmarks comparing A1, A2, and A3 implementations"""
    results: List[Dict] = []
    
    print("=== Assignment 3: A1 vs A2 vs A3 Benchmarks ===")
    
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    
    for N in cfg.sizes:
        print(f"\nBenchmarking N={N}:")
        
        # A1: Sequential baseline
        print("  Running A1: Sequential...")
        t_seq = jacobi_sequential_run(N, cfg.iters)
        results.append({
            "N": N, 
            "type": "sequential", 
            "time_sec": t_seq, 
            "speedup_vs_seq": 1.0,
            "warmup_time": 0.0
        })
        
        # A2: Multiprocessing
        print("  Running A2: Multiprocessing...")
        t_mp = jacobi_multiprocessing_run(N, cfg.iters, W=4)
        speedup_mp = t_seq / t_mp if t_mp > 0 else float("inf")
        results.append({
            "N": N,
            "type": "multiprocessing",
            "time_sec": t_mp,
            "speedup_vs_seq": speedup_mp,
            "warmup_time": 0.0
        })
        
        # A3: Numba parallel  
        print("  Running A3: Numba parallel...")
        t_numba, warmup_time = jacobi_numba_run(N, cfg.iters, cfg.warmup_iters)
        speedup_numba = t_seq / t_numba if t_numba > 0 else float("inf")
        results.append({
            "N": N,
            "type": "numba_parallel",
            "time_sec": t_numba,
            "speedup_vs_seq": speedup_numba,
            "warmup_time": warmup_time
        })
        
        print(f"  A1 Sequential:      {t_seq:.3f}s")
        print(f"  A2 Multiprocessing: {t_mp:.3f}s ({speedup_mp:.2f}× speedup)")
        print(f"  A3 Numba:           {t_numba:.3f}s ({speedup_numba:.2f}× speedup, warmup: {warmup_time:.3f}s)")
    
    return results

# =============================
# CLI / Main
# =============================
def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Assignment 3 — Numba OpenMP-Style Parallel Jacobi Solver")
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024],
                   help="Grid sizes N")
    p.add_argument("--iters", type=int, default=500,
                   help="Number of Jacobi iterations T (default 500)")
    p.add_argument("--outdir", type=str, default="out_a3",
                   help="Output directory (csv/plots/report)")
    p.add_argument("--warmup", type=int, default=5,
                   help="JIT warmup iterations (default 5)")
    p.add_argument("--no-pdf", action="store_true",
                   help="Disable PDF conversion via md-to-pdf CLI")
    p.add_argument("--pdf-timeout", type=int, default=30,
                   help="Timeout for md-to-pdf conversion (seconds)")
    args = p.parse_args()
    
    return RunConfig(
        sizes=tuple(args.sizes),
        iters=int(args.iters),
        outdir=args.outdir,
        warmup_iters=int(args.warmup),
        make_pdf=(not args.no_pdf),
        pdf_timeout=int(args.pdf_timeout),
    )

def main():
    cfg = parse_args()
    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = run_benchmarks(cfg)
    
    # Save and visualize results
    save_csv(results, os.path.join(cfg.outdir, "runtime.csv"))
    plot_runtime_comparison(results, os.path.join(cfg.outdir, "runtime_comparison.png"))
    plot_speedup_analysis(results, os.path.join(cfg.outdir, "speedup_analysis.png"))
    write_report_md(results, cfg.iters, cfg.outdir, cfg.make_pdf, cfg.pdf_timeout)
    
    # Console summary
    print("\n=== Final Results Summary ===")
    Ns = sorted({r["N"] for r in results})
    for N in Ns:
        seq = next(r for r in results if r["N"] == N and r["type"] == "sequential")
        mp = next(r for r in results if r["N"] == N and r["type"] == "multiprocessing")
        numba = next(r for r in results if r["N"] == N and r["type"] == "numba_parallel")
        print(f"N={N:4d}:")
        print(f"  A1 Sequential:      {seq['time_sec']:.3f}s (baseline)")
        print(f"  A2 Multiprocessing: {mp['time_sec']:.3f}s ({mp['speedup_vs_seq']:.1f}× speedup)")
        print(f"  A3 Numba Parallel:  {numba['time_sec']:.3f}s ({numba['speedup_vs_seq']:.1f}× speedup)")
        print()

if __name__ == "__main__":
    main()
