from __future__ import annotations
import argparse, os, csv, time, math, shutil, subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from pathlib import Path

# =============================
# Config & CLI
# =============================
@dataclass
class RunConfig:
    sizes: Tuple[int, ...]           # N ∈ {256,512,1024}
    iters: int                       # T = 500 per assignment
    workers: Tuple[int, ...]         # W ∈ {2,4,8}
    outdir: str                      # output directory
    make_pdf: bool = True            # convert MD -> PDF via md-to-pdf CLI
    pdf_timeout: int = 30            # seconds

# =============================
# Core: Dirichlet grid init
# Top boundary = 100, others = 0
# =============================
def init_dirichlet_grid(N: int) -> Tuple[np.ndarray, np.ndarray]:
    u = np.zeros((N, N), dtype=np.float64)
    u_new = np.zeros_like(u)
    u[0, :] = 100.0
    u_new[0, :] = 100.0
    return u, u_new

# =============================
# Partition interior rows [1..N-2]
# Contiguous, balanced ±1 row
# =============================
def partition_rows(N: int, W: int) -> List[Tuple[int, int]]:
    start = 1
    end = N - 1
    total = end - start  # N-2 interior rows
    base = total // W
    rem = total % W
    parts = []
    cur = start
    for i in range(W):
        add = base + (1 if i < rem else 0)
        parts.append((cur, cur + add))  # [r0, r1)
        cur += add
    return parts

# =============================
# Worker globals (shared memory)
# =============================
_g_shape: Tuple[int, int] | None = None
_g_u: np.ndarray | None = None       # shared input grid u (authoritative copy)
_g_un: np.ndarray | None = None      # shared output grid u_new
_g_chunks: Dict[int, np.ndarray] = {}  # per-chunk halo buffers (H_i x N), H_i = (rows_i + 2)

# IMPORTANT: keep SharedMemory object references alive in worker (macOS fix)
_g_main_shms: List[shared_memory.SharedMemory] = []   # [shm_u, shm_un]
_g_chunk_shms: List[shared_memory.SharedMemory] = []  # [shm_chunk_0, shm_chunk_1, ...]

def worker_initializer(
    u_name: str,
    un_name: str,
    shape: Tuple[int, int],
    chunk_names: List[str],
    chunk_shapes: List[Tuple[int, int]]
):
    """Attach worker to shared u/u_new and to per-chunk halo buffers.
       IMPORTANT: Keep SharedMemory objects referenced to avoid premature close on macOS."""
    global _g_shape, _g_u, _g_un, _g_chunks, _g_main_shms, _g_chunk_shms
    _g_shape = shape

    # Main shared arrays (keep shm objects in a global list)
    shm_u = shared_memory.SharedMemory(name=u_name)
    shm_un = shared_memory.SharedMemory(name=un_name)
    _g_main_shms = [shm_u, shm_un]   # <-- keep alive
    _g_u = np.ndarray(shape, dtype=np.float64, buffer=shm_u.buf)
    _g_un = np.ndarray(shape, dtype=np.float64, buffer=shm_un.buf)

    # Halo shared arrays (one per chunk) — keep shm objects alive too
    _g_chunks = {}
    _g_chunk_shms = []               # <-- keep alive
    for i, nm in enumerate(chunk_names):
        shm = shared_memory.SharedMemory(name=nm)
        _g_chunk_shms.append(shm)    # <-- keep alive
        _g_chunks[i] = np.ndarray(chunk_shapes[i], dtype=np.float64, buffer=shm.buf)

def jacobi_chunk_with_halo(args: Tuple[int, int, int]):
    """
    Worker computes Jacobi update for its subrange [r0:r1) using the explicit halo buffer.
    args: (chunk_idx, r0, r1)
    The halo buffer shape is H x N, where H = (r1 - r0) + 2.
      halo[1:-1, :]   -> rows r0..r1-1 (local interior)
      halo[0,   :]    -> row r0-1 (top neighbor)
      halo[-1,  :]    -> row r1   (bottom neighbor)
    Writes into shared u_new for columns 1..N-2; parent will reapply fixed boundaries.
    """
    global _g_un, _g_chunks, _g_shape
    idx, r0, r1 = args
    halo = _g_chunks[idx]
    N, _ = _g_shape

    # Compute interior columns 1..N-2 using halo neighbors
    # Map halo rows: for output rows r0..r1-1 -> halo rows 1..H-2
    _g_un[r0:r1, 1:N-1] = 0.25 * (
        halo[0:-2, 1:N-1] +     # up    (k-1)
        halo[2:  , 1:N-1] +     # down  (k+1)
        halo[1:-1, 0:N-2] +     # left
        halo[1:-1, 2:  ]        # right
    )
    # no return

# =============================
# Sequential baseline (W=1)
# =============================
def jacobi_sequential_run(N: int, T: int) -> float:
    u, u_new = init_dirichlet_grid(N)
    t0 = time.perf_counter()
    for _ in range(T):
        u_new[1:-1, 1:-1] = 0.25 * (
            u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
        )
        # fixed boundaries
        u_new[0, :] = 100.0
        u_new[-1, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0
        u, u_new = u_new, u
    t1 = time.perf_counter()
    return t1 - t0

# =============================
# Parallel run (explicit halo exchange)
# =============================
def jacobi_parallel_run(N: int, T: int, W: int, warmup: int = 1) -> float:
    """
    Explicit halo-exchange variant:
      - Parent maintains authoritative shared input grid u_sh and output grid un_sh.
      - For each chunk i, parent allocates a persistent shared halo buffer of shape ((rows_i+2) x N).
      - Each iteration:
          * Parent copies r0-1..r1 rows from u_sh into the chunk's halo buffer
            (with clamping at physical boundaries; top row already fixed to 100).
          * Workers compute using their halo buffer -> write into un_sh[r0:r1, :]
          * Parent reapplies fixed boundaries and swaps u_sh/un_sh.
    """
    # Base arrays
    u, u_new = init_dirichlet_grid(N)

    # Shared main grids
    shm_u = shared_memory.SharedMemory(create=True, size=u.nbytes)
    shm_un = shared_memory.SharedMemory(create=True, size=u_new.nbytes)
    u_sh  = np.ndarray(u.shape,    dtype=u.dtype,    buffer=shm_u.buf)
    un_sh = np.ndarray(u_new.shape, dtype=u_new.dtype, buffer=shm_un.buf)
    u_sh[:] = u
    un_sh[:] = u_new

    # Partition and create per-chunk halo shared buffers
    parts = partition_rows(N, W)
    chunk_shms: List[shared_memory.SharedMemory] = []
    chunk_shapes: List[Tuple[int, int]] = []
    for (r0, r1) in parts:
        H = (r1 - r0) + 2
        shm = shared_memory.SharedMemory(create=True, size=(H * N * np.float64().nbytes))
        chunk_shms.append(shm)
        chunk_shapes.append((H, N))
        # Initialize (optional)
        np.ndarray((H, N), dtype=np.float64, buffer=shm.buf).fill(0.0)

    # Prepare names for initializer
    chunk_names = [shm.name for shm in chunk_shms]

    try:
        with ProcessPoolExecutor(
            max_workers=W,
            initializer=worker_initializer,
            initargs=(shm_u.name, shm_un.name, u.shape, chunk_names, chunk_shapes)
        ) as pool:

            def fill_halos_from_u():
                """Parent-side explicit boundary exchange into each chunk's halo buffer."""
                for i, (r0, r1) in enumerate(parts):
                    H, _ = chunk_shapes[i]
                    halo = np.ndarray(chunk_shapes[i], dtype=np.float64, buffer=chunk_shms[i].buf)
                    # center rows
                    halo[1:-1, :] = u_sh[r0:r1, :]
                    # top neighbor row (r0-1) or clamp to physical boundary row 0
                    top_src = r0 - 1 if r0 - 1 >= 0 else 0
                    halo[0, :] = u_sh[top_src, :]
                    # bottom neighbor row (r1) or clamp to last row
                    bottom_src = r1 if r1 < N else N - 1
                    halo[-1, :] = u_sh[bottom_src, :]

            # Warmup (not timed)
            for _ in range(max(0, warmup)):
                fill_halos_from_u()
                futures = [pool.submit(jacobi_chunk_with_halo, (i, r0, r1))
                           for i, (r0, r1) in enumerate(parts)]
                wait(futures, return_when=ALL_COMPLETED)
                # fixed boundaries on un_sh
                un_sh[0, :] = 100.0; un_sh[-1, :] = 0.0; un_sh[:, 0] = 0.0; un_sh[:, -1] = 0.0
                # swap
                u_sh[:], un_sh[:] = un_sh, u_sh

            # Timed iterations
            t0 = time.perf_counter()
            for _ in range(T):
                fill_halos_from_u()
                futures = [pool.submit(jacobi_chunk_with_halo, (i, r0, r1))
                           for i, (r0, r1) in enumerate(parts)]
                wait(futures, return_when=ALL_COMPLETED)
                # fixed boundaries
                un_sh[0, :] = 100.0; un_sh[-1, :] = 0.0; un_sh[:, 0] = 0.0; un_sh[:, -1] = 0.0
                # swap for next iter
                u_sh[:], un_sh[:] = un_sh, u_sh
            t1 = time.perf_counter()

    finally:
        # cleanup shared memories
        shm_u.close(); shm_un.close()
        shm_u.unlink(); shm_un.unlink()
        for shm in chunk_shms:
            try:
                shm.close(); shm.unlink()
            except Exception:
                pass

    return t1 - t0

# =============================
# Helpers: CSV / plotting / report (md-to-pdf)
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

def plot_runtime(rows: List[Dict], path: str) -> None:
    ensure_outdir(os.path.dirname(path))
    plt.figure(figsize=(7.5, 4.5))
    Ns = sorted({r["N"] for r in rows})
    for N in Ns:
        sub = [r for r in rows if r["N"] == N]
        sub = sorted(sub, key=lambda x: x["W"])
        xs = [r["W"] for r in sub]
        ys = [r["time_sec"] for r in sub]
        plt.plot(xs, ys, marker="o", label=f"N={N}")
    plt.xlabel("Workers (W)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Jacobi Heat Diffusion: Runtime vs Workers (Explicit Halo Exchange)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def build_interpretation(runtime_rows: List[Dict]) -> str:
    """
    Auto-generates a concise interpretation section from measured results.
    It comments on (a) small vs large N behavior, (b) best observed speedup,
    and (c) why overheads dominate or are amortized.
    """
    # Organize by N
    byN: Dict[int, List[Dict]] = {}
    for r in runtime_rows:
        byN.setdefault(r["N"], []).append(r)

    lines: List[str] = []
    best_speedup = 0.0
    best_tuple = None  # (N, W, speedup)

    for N, rows in sorted(byN.items()):
        rows = sorted(rows, key=lambda x: x["W"])
        # find seq and best parallel
        seq = next((x for x in rows if x["W"] == 1), None)
        par = [x for x in rows if x["W"] != 1]
        if not seq or not par:
            continue

        # best speedup for this N
        local_best = max(par, key=lambda x: x["speedup_vs_seq"])
        if local_best["speedup_vs_seq"] > best_speedup:
            best_speedup = local_best["speedup_vs_seq"]
            best_tuple = (N, local_best["W"], best_speedup)

        # summarize shape for this N
        worst = min(par, key=lambda x: x["speedup_vs_seq"])
        trend = "improves with more workers" if par[-1]["time_sec"] < par[0]["time_sec"] else "degrades as workers increase"

        lines.append(
            f"- **N={N}**: sequential = {seq['time_sec']:.3f}s. "
            f"Best parallel at W={local_best['W']} → {local_best['time_sec']:.3f}s "
            f"({local_best['speedup_vs_seq']:.2f}×). "
            f"Worst at W={worst['W']} → {worst['time_sec']:.3f}s "
            f"({worst['speedup_vs_seq']:.2f}×). Overall, runtime {trend}."
        )

    # High-level summary
    if best_tuple:
        Nb, Wb, Sb = best_tuple
        lines.insert(
            0,
            f"Across all sizes, the **best observed speedup** is **{Sb:.2f}×** "
            f"at **N={Nb}, W={Wb}**."
        )

    # Qualitative reasoning
    lines.append("")
    lines.append(
        "- For smaller grids, the computation per worker is too small to hide overheads "
        "(process synchronization, halo copies, and per-iteration barriers), so speedup < 1."
    )
    lines.append(
        "- As N grows, each iteration performs more floating-point work per chunk while the halo "
        "copy remains proportional to the boundary length, so the **compute/communication ratio improves**. "
        "This yields modest positive scaling when N is large enough."
    )
    lines.append(
        "- Synchronization is still required each iteration to keep Jacobi consistent; this barrier "
        "introduces idle time and limits scaling even for larger N."
    )

    return "\n".join(lines)


def write_report_md(runtime_rows: List[Dict], iters: int, outdir: str, make_pdf: bool, pdf_timeout: int):
    os.makedirs(outdir, exist_ok=True)
    md_path = os.path.join(outdir, "report.md")

    # Build results table
    headers = ["N", "W", "time_sec", "speedup_vs_seq"]
    lines_table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    Ns = sorted({r["N"] for r in runtime_rows})
    for N in Ns:
        sub = sorted([r for r in runtime_rows if r["N"] == N], key=lambda x: x["W"])
        for r in sub:
            lines_table.append(
                f"| {r['N']} | {r['W']} | {r['time_sec']:.3f} | {r['speedup_vs_seq']:.2f}x |"
            )

    interpretation = build_interpretation(runtime_rows)

    md_content = [
        "# Assignment 2 – Threads & Synchronization (Processes)",
        "",
        f"**Jacobi Iteration on 2D Grid (T = {iters})**",
        "",
        "## Method & Partitioning",
        "- Dirichlet boundaries: top = 100, others = 0.",
        "- Parallelism via **`ProcessPoolExecutor` + `multiprocessing.shared_memory`**.",
        "- **Row-wise block partitioning** of interior rows `[1..N-2]` with near-balanced chunks.",
        "- Each worker writes only to its own row range in `u_new`, eliminating write conflicts.",
        "- **Explicit halo exchange**: parent copies the neighbor rows into per-chunk halo buffers each iteration.",
        "",
        "## Synchronization & Race Avoidance",
        "- **Per-iteration barrier** (`wait(...)`) ensures all workers finish before boundaries are reapplied and buffers are swapped.",
        "- Parent **reapplies Dirichlet boundaries** after each iteration and **swaps** `u`/`u_new`.",
        "- **No races** by construction: writes are disjoint, reads come from the worker’s halo buffer.",
        "",
        "## Runtime vs Workers",
        "![runtime plot](runtime.png)",
        "",
        "## Results Table",
        *lines_table,
        "",
        "## Interpretation",
        interpretation,
        "",
        "## Overheads",
        "- Process pool startup & synchronization (mitigated by reusing one pool).",
        "- Parent-side halo copying each iteration (contiguous row copies).",
        "- Per-iteration barrier induces idle time across workers.",
        "- Shared memory **avoids pickling** large arrays, reducing transfer costs.",
        "",
        "## Conclusion",
        "Parallel Jacobi requires halo exchange and iteration-level synchronization to guarantee correctness. "
        "For small N, these overheads dominate and cause slowdowns. For larger N, the increased compute per chunk "
        "amortizes communication and synchronization, yielding modest speedup. Overall, scaling is limited by the "
        "per-iteration barrier and the surface-to-volume trade-off inherent to stencil computations.",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"✅ Wrote {md_path}")

    # Convert to PDF via CLI `md-to-pdf` if available
    cli = shutil.which("md-to-pdf")
    if make_pdf and cli is not None:
        try:
            subprocess.run(
                ["md-to-pdf", "report.md"],
                cwd=outdir,
                check=True,
                timeout=pdf_timeout,
            )
            print(f"✅ PDF created at {os.path.join(outdir,'report.pdf')}")
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
    results: List[Dict] = []
    for N in cfg.sizes:
        # sequential
        t_seq = jacobi_sequential_run(N, cfg.iters)
        results.append({"N": N, "W": 1, "time_sec": t_seq, "speedup_vs_seq": 1.0})
        # parallel with explicit halo exchange
        for W in cfg.workers:
            t_par = jacobi_parallel_run(N, cfg.iters, W)
            speedup = (t_seq / t_par) if t_par > 0 else float("inf")
            results.append({"N": N, "W": W, "time_sec": t_par, "speedup_vs_seq": speedup})
    return results

# =============================
# CLI / Main
# =============================
def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Assignment 2 — Jacobi with Process-Based Parallelism (Explicit Halo Exchange)")
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024],
                   help="Grid sizes N")
    p.add_argument("--iters", type=int, default=500,
                   help="Number of Jacobi iterations T (default 500)")
    p.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8],
                   help="Worker counts W")
    p.add_argument("--outdir", type=str, default="out_a2",
                   help="Output directory (csv/plots/report)")
    p.add_argument("--no-pdf", action="store_true",
                   help="Disable PDF conversion via md-to-pdf CLI")
    p.add_argument("--pdf-timeout", type=int, default=30,
                   help="Timeout for md-to-pdf conversion (seconds)")
    args = p.parse_args()
    return RunConfig(
        sizes=tuple(args.sizes),
        iters=int(args.iters),
        workers=tuple(args.workers),
        outdir=args.outdir,
        make_pdf=(not args.no_pdf),
        pdf_timeout=int(args.pdf_timeout),
    )

def main():
    mp.set_start_method("spawn", force=True)  # robust for macOS/Windows
    cfg = parse_args()
    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)

    results = run_benchmarks(cfg)

    save_csv(results, os.path.join(cfg.outdir, "runtime.csv"))
    plot_runtime(results, os.path.join(cfg.outdir, "runtime.png"))
    write_report_md(results, cfg.iters, cfg.outdir, cfg.make_pdf, cfg.pdf_timeout)

    # Pretty console summary
    print("\n=== Results (by N) ===")
    Ns = sorted({r["N"] for r in results})
    for N in Ns:
        sub = sorted([r for r in results if r["N"] == N], key=lambda x: x["W"])
        for r in sub:
            print(f"N={N:4d}, W={r['W']:2d} -> {r['time_sec']:.3f}s, speedup {r['speedup_vs_seq']:.2f}x")

if __name__ == "__main__":
    main()
