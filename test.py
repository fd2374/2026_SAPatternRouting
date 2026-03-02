"""
Multi-seed parallel benchmark runner.

Runs the SA router with multiple random seeds in parallel (one process per seed),
then reports the best result and a summary table across all seeds.

Usage:
  python test.py benchmarks/dense1
  python test.py benchmarks/dense2 --seeds 10 --workers 4
  python test.py benchmarks/dense1 --beta 20 --max-iter 1000
"""

import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from router.parser import load_benchmark
from router.solver import solve_sa, generate_all_candidates
from router.util import compute_max_coupling_distance
from router.output import write_solution
from router.visualizer import visualize_solution


def _run_one_seed(benchmark_dir, alpha, beta, gamma, max_iter, seed,
                  candidates_pickle, package_pickle, max_cd):
    """Worker function executed in a child process."""
    import pickle
    random.seed(seed)
    package = pickle.loads(package_pickle)
    candidates = pickle.loads(candidates_pickle)

    t0 = time.time()
    sol = solve_sa(
        package, candidates,
        alpha, beta, gamma, max_iter,
        max_coupling_distance=max_cd, verbose=False,
    )
    elapsed = time.time() - t0

    return {
        "seed": seed,
        "wirelength": sol.total_wirelength,
        "crosstalk": sol.total_crosstalk,
        "max_xt": sol.max_crosstalk,
        "vias": sol.num_vias,
        "crossings": sol.num_crossings,
        "drc": sol.num_drc_violations,
        "time": elapsed,
        "n_routes": len(sol.routes),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-seed parallel SA runner")
    parser.add_argument("benchmark", help="Path to benchmark directory")
    parser.add_argument("--seeds", type=int, default=8,
                        help="Number of random seeds to try (default: 8)")
    parser.add_argument("--seed-start", type=int, default=1,
                        help="First seed value (default: 1)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: CPU count)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization for the best result")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    benchmark_dir = os.path.abspath(args.benchmark)
    benchmark_name = os.path.basename(os.path.normpath(benchmark_dir))

    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "results", benchmark_name)

    print(f"{'=' * 70}")
    print(f"  Multi-Seed Parallel Router")
    print(f"  Benchmark : {benchmark_name}")
    print(f"  Seeds     : {args.seeds}  (start={args.seed_start})")
    print(f"  Workers   : {args.workers or 'auto'}")
    print(f"  Weights   : alpha={args.alpha}  beta={args.beta}  gamma={args.gamma}")
    print(f"  Max iter  : {args.max_iter}")
    print(f"{'=' * 70}\n")

    print("[Test] Loading benchmark...")
    package = load_benchmark(benchmark_dir)
    print(f"[Test] {len(package.nets)} nets, {package.params.num_layers} layers\n")

    print("[Test] Generating candidates (shared across all seeds)...")
    t0 = time.time()
    candidates = generate_all_candidates(package, verbose=False)
    total_cands = sum(len(c) for c in candidates)
    print(f"[Test] {total_cands} candidates in {time.time() - t0:.2f}s\n")

    max_cd = compute_max_coupling_distance(package.params)

    import pickle
    pkg_pkl = pickle.dumps(package)
    cand_pkl = pickle.dumps(candidates)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    results = []

    print(f"[Test] Launching {len(seeds)} SA runs in parallel...\n")
    wall_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_one_seed, benchmark_dir,
                args.alpha, args.beta, args.gamma, args.max_iter,
                s, cand_pkl, pkg_pkl, max_cd,
            ): s
            for s in seeds
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"  seed={r['seed']:<4d}  WL={r['wirelength']:.1f}  "
                  f"XT={r['crosstalk']:.2f}  Vias={r['vias']}  "
                  f"Xings={r['crossings']}  DRC={r['drc']}  "
                  f"t={r['time']:.2f}s")

    wall_elapsed = time.time() - wall_start
    results.sort(key=lambda r: r["seed"])

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY TABLE")
    print(f"  {'Seed':>6s}  {'WL':>10s}  {'XT':>10s}  {'MaxXT':>10s}  "
          f"{'Vias':>5s}  {'Xings':>5s}  {'DRC':>5s}  {'Time':>6s}")
    print(f"  {'-' * 65}")
    for r in results:
        print(f"  {r['seed']:>6d}  {r['wirelength']:>10.1f}  {r['crosstalk']:>10.2f}  "
              f"{r['max_xt']:>10.2f}  {r['vias']:>5d}  {r['crossings']:>5d}  "
              f"{r['drc']:>5d}  {r['time']:>5.1f}s")
    print(f"  {'-' * 65}")

    wls = [r["wirelength"] for r in results]
    xts = [r["crosstalk"] for r in results]
    xings = [r["crossings"] for r in results]
    drcs = [r["drc"] for r in results]
    print(f"  {'AVG':>6s}  {sum(wls)/len(wls):>10.1f}  {sum(xts)/len(xts):>10.2f}  "
          f"{'':>10s}  {'':>5s}  {sum(xings)/len(xings):>5.1f}  "
          f"{sum(drcs)/len(drcs):>5.1f}")
    print(f"  {'MIN':>6s}  {min(wls):>10.1f}  {min(xts):>10.2f}  "
          f"{'':>10s}  {'':>5s}  {min(xings):>5d}  {min(drcs):>5d}")
    print(f"  {'MAX':>6s}  {max(wls):>10.1f}  {max(xts):>10.2f}  "
          f"{'':>10s}  {'':>5s}  {max(xings):>5d}  {max(drcs):>5d}")

    best = min(results, key=lambda r: (
        r["crossings"] + r["drc"], r["crosstalk"], r["wirelength"]
    ))
    print(f"\n  BEST seed = {best['seed']}  "
          f"(Xings+DRC={best['crossings']+best['drc']}, "
          f"XT={best['crosstalk']:.2f}, WL={best['wirelength']:.1f})")
    print(f"  Wall time = {wall_elapsed:.2f}s")
    print(f"{'=' * 70}\n")

    print("[Test] Re-running best seed to save results...")
    random.seed(best["seed"])
    solution = solve_sa(
        package, candidates,
        args.alpha, args.beta, args.gamma, args.max_iter,
        max_coupling_distance=max_cd, verbose=True,
    )
    write_solution(solution, output_dir)

    if not args.no_viz:
        print("\n[Test] Generating visualizations for best result...")
        visualize_solution(solution, output_dir, show=not args.no_show)

    print(f"\n[Test] Done! Best results (seed={best['seed']}) in {output_dir}/")


if __name__ == "__main__":
    main()
