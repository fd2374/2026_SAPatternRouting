"""
Crosstalk-Aware Package-Level Pattern Router
Main entry point.

Usage:
  python -m router.main <benchmark_dir> [options]

Examples:
  python -m router.main benchmarks/dense2
  python -m router.main benchmarks/pkg1 --alpha 1 --beta 5 --gamma 3
"""

import argparse
import os
import random
import time

from .parser import load_benchmark
from .solver import solve_sa, generate_all_candidates
from .util import compute_max_coupling_distance
from .output import write_solution
from .visualizer import visualize_solution
from .postprocess import greedy_remove, astar_reroute


def _print_comparison(baseline, solution, t_base, t_solve):
    """Print side-by-side comparison of baseline vs aware solution."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON (Unaware vs Aware):")
    print(f"  {'Metric':<25s} {'Unaware':>12s} {'Aware':>12s} {'Improve':>10s}")
    print(f"  {'-'*59}")
    for label, v0, v1 in [
        ("Total wirelength", baseline.total_wirelength, solution.total_wirelength),
        ("Total crosstalk", baseline.total_crosstalk, solution.total_crosstalk),
        ("Max pair crosstalk", baseline.max_crosstalk, solution.max_crosstalk),
        ("Total vias", baseline.num_vias, solution.num_vias),
        ("Same-layer crossings", baseline.num_crossings, solution.num_crossings),
        ("DRC violations", baseline.num_drc_violations, solution.num_drc_violations),
    ]:
        if v0 > 1e-9:
            pct = (v0 - v1) / v0 * 100
            print(f"  {label:<25s} {v0:>12.2f} {v1:>12.2f} {pct:>+9.1f}%")
        else:
            print(f"  {label:<25s} {v0:>12.2f} {v1:>12.2f}       N/A")
    print(f"  {'-'*59}")
    print(f"  Solve time:  baseline={t_base:.2f}s  aware={t_solve:.2f}s")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Crosstalk-Aware Package-Level Pattern Router (135-degree)")
    parser.add_argument("benchmark", help="Path to benchmark directory")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Wirelength weight (default: 1.0)")
    parser.add_argument("--beta", type=float, default=10.0,
                        help="Crosstalk weight (default: 10.0)")
    parser.add_argument("--gamma", type=float, default=0,
                        help="Via count weight (default: 0)")
    parser.add_argument("--max-iter", type=int, default=5000,
                        help="Max SA temperature loops (default: 5000)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: results/<benchmark_name>)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots (only save)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    benchmark_dir = os.path.abspath(args.benchmark)
    benchmark_name = os.path.basename(os.path.normpath(benchmark_dir))

    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", benchmark_name)

    print(f"{'='*60}")
    print(f" Crosstalk-Aware Package Router")
    print(f" Benchmark: {benchmark_name}")
    print(f" Weights: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    print(f" Seed: {args.seed}")
    print(f"{'='*60}\n")

    random.seed(args.seed)

    # Load benchmark
    print("[Main] Loading benchmark...")
    package = load_benchmark(benchmark_dir)
    print(f"[Main] Package: {package.params.pkg_width} x {package.params.pkg_height}")
    print(f"[Main] Layers: {package.params.num_layers}")
    print(f"[Main] Dies: {len(package.dies)}")
    print(f"[Main] Nets: {len(package.nets)}")
    print(f"[Main] Bumps: {len(package.bumps)}")
    print()

    # Generate candidates (shared between baseline and main)
    print("[Main] Generating candidates...")
    t0 = time.time()
    candidates = generate_all_candidates(package, verbose=True)
    total_cands = sum(len(c) for c in candidates)
    print(f"[Main] Generated {total_cands} total candidates in {time.time()-t0:.2f}s")

    max_cd = compute_max_coupling_distance(package.params)

    # Main: crosstalk-aware
    print(f"\n{'='*60}")
    print(f"[Main] Running crosstalk-AWARE routing (beta={args.beta})...")
    t0 = time.time()
    solution = solve_sa(
        package, candidates,
        args.alpha, args.beta, args.gamma, args.max_iter,
        max_coupling_distance=max_cd, verbose=True,
        live_plot=not args.no_viz,
    )
    t_solve = time.time() - t0

    # Post-processing: greedy removal + A* rerouting
    if solution.num_crossings + solution.num_drc_violations > 0:
        print(f"\n{'='*60}")
        print("[Post] Greedy removal of violating nets...")
        t_post = time.time()
        _, removed_nets = greedy_remove(solution, verbose=True)
        if removed_nets:
            print(f"\n[Post] A* rerouting {len(removed_nets)} nets...")
            solution = astar_reroute(solution, removed_nets, verbose=True)
            print(f"  [Post] After reroute: Xings={solution.num_crossings} "
                  f"DRC={solution.num_drc_violations}")
        t_post = time.time() - t_post
        print(f"  [Post] Post-processing time: {t_post:.2f}s")
    else:
        removed_nets = []
        print("\n[Post] No violations, skipping post-processing.")
        t_post = 0.0

    print(f"\n[Main] Total solve time: {t_solve + t_post:.2f}s")

    # Write output
    write_solution(solution, output_dir)

    # Visualization
    if not args.no_viz:
        print("\n[Main] Generating visualizations...")
        visualize_solution(solution, output_dir, show=not args.no_show)

    print(f"\n[Main] Done! Results in {output_dir}/")


if __name__ == "__main__":
    main()
