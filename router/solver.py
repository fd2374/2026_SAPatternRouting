"""
Route selection solver: Simulated Annealing.

Given N nets, each with K candidate routing patterns, select one pattern
per net to minimize:
  cost = alpha * wirelength + beta * crosstalk + gamma * vias
       + CROSSING_PENALTY * crossings + DRC_PENALTY * drc_violations

Uses single-net perturbation per step for O(n) incremental delta.
"""

import math
import random
from functools import partial
from typing import List

from .data_types import RoutePattern, Package, RoutingSolution
from .util import (
    total_crosstalk, max_pair_crosstalk,
    compute_max_coupling_distance, total_crossings,
    get_via_obstacles, total_drc_violations,
    extract_seg_array, extract_obs_array, pair_term_jit,
)
from .pattern_gen import generate_candidates

CROSSING_PENALTY = 1e6
DRC_PENALTY = 1e6


def solve_sa(
    package: Package,
    candidates: List[List[RoutePattern]],
    alpha: float = 1.0,
    beta: float = 10.0,
    gamma: float = 5.0,
    max_iters: int = 500,
    max_coupling_distance: float = None,
    verbose: bool = True,
    live_plot: bool = False,
) -> RoutingSolution:
    """Simulated Annealing solver with incremental delta evaluation."""
    params = package.params
    nets = package.nets
    n = len(nets)

    if max_coupling_distance is None:
        max_coupling_distance = compute_max_coupling_distance(params)

    if verbose:
        print(f"[SA] {n} nets, max_coupling_dist={max_coupling_distance:.1f}")

    if n == 0:
        return RoutingSolution(package=package, selected=[], candidates=candidates)

    # Pre-extract numpy arrays for all candidates (once)
    via_obs_all = [[get_via_obstacles(c, nets[i], params)
                    for c in candidates[i]] for i in range(n)]
    seg_arrs = [[extract_seg_array(c) for c in candidates[i]]
                for i in range(n)]
    obs_arrs = [[extract_obs_array(via_obs_all[i][k])
                 for k in range(len(candidates[i]))] for i in range(n)]

    pt = partial(pair_term_jit,
                 params.wire_width, params.min_center_distance,
                 max_coupling_distance, beta, CROSSING_PENALTY, DRC_PENALTY)

    # --- Random initialization ---
    selected = [random.randrange(len(candidates[i])) for i in range(n)]
    routes = [candidates[i][selected[i]] for i in range(n)]
    cur_seg = [seg_arrs[i][selected[i]] for i in range(n)]
    cur_obs = [obs_arrs[i][selected[i]] for i in range(n)]

    def total_energy():
        wl = sum(r.wirelength for r in routes)
        nv = sum(r.num_vias for r in routes)
        pair = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                pair += pt(cur_seg[i], cur_seg[j], cur_obs[i], cur_obs[j])
        return alpha * wl + gamma * nv + pair

    energy = total_energy()
    best_energy = energy
    best_selected = list(selected)

    # --- Temperature schedule ---
    # T0 from average unary cost range (avoids penalty contamination)
    # t0 = _init_temperature(n, candidates, alpha, gamma)
    t0 = 1e7
    cool_rate = 0.995
    temperature = t0
    steps_per_iter = max(20, n * 3)

    if verbose:
        print(f"[SA] T0={t0:.2f}, cool_rate={cool_rate}, "
              f"steps/iter={steps_per_iter}, iters={max_iters}")
        _print_stats("init", routes, nets, params, max_coupling_distance)

    # --- Live plot ---
    sa_plot = None
    if live_plot:
        from .visualizer import SALivePlot
        sa_plot = SALivePlot()

    # --- Main loop ---
    for t in range(max_iters):
        accepted = 0

        for _ in range(steps_per_iter):
            i = random.randrange(n)
            if len(candidates[i]) <= 1:
                continue
            new_k = random.randrange(len(candidates[i]))
            if new_k == selected[i]:
                continue

            new_route = candidates[i][new_k]
            old_route = routes[i]
            new_sa = seg_arrs[i][new_k]
            new_oa = obs_arrs[i][new_k]

            # O(n) incremental delta
            delta = alpha * (new_route.wirelength - old_route.wirelength)
            delta += gamma * (new_route.num_vias - old_route.num_vias)
            for j in range(n):
                if j == i:
                    continue
                delta += (pt(new_sa, cur_seg[j], new_oa, cur_obs[j]) -
                          pt(cur_seg[i], cur_seg[j], cur_obs[i], cur_obs[j]))

            if delta <= 0 or random.random() < math.exp(-delta / max(temperature, 1e-9)):
                selected[i] = new_k
                routes[i] = new_route
                cur_seg[i] = new_sa
                cur_obs[i] = new_oa
                energy += delta
                accepted += 1
                if energy < best_energy:
                    best_energy = energy
                    best_selected = list(selected)

        temperature *= cool_rate

        if sa_plot is not None:
            sa_plot.update(t + 1, energy, best_energy,
                           temperature, accepted / steps_per_iter)

        if verbose and (t < 3 or (t + 1) % 100 == 0 or t == max_iters - 1):
            _print_stats(f"iter {t+1}", routes, nets, params,
                         max_coupling_distance)
            print(f"  T={temperature:.3f}  acc={accepted}/{steps_per_iter}  "
                  f"E={energy:.1f}  E*={best_energy:.1f}")

    if sa_plot is not None:
        sa_plot.finish()

    # Restore best
    selected = best_selected
    routes = [candidates[i][selected[i]] for i in range(n)]

    if verbose:
        _print_stats("best", routes, nets, params, max_coupling_distance)

    return RoutingSolution(
        package=package,
        selected=selected,
        candidates=candidates,
        total_wirelength=sum(r.wirelength for r in routes),
        total_crosstalk=total_crosstalk(routes, params, max_coupling_distance),
        max_crosstalk=max_pair_crosstalk(routes, params, max_coupling_distance),
        num_vias=sum(r.num_vias for r in routes),
        num_crossings=total_crossings(routes),
        num_drc_violations=total_drc_violations(routes, nets, params),
    )


def _init_temperature(n, candidates, alpha, gamma):
    """Estimate T0 from average unary cost range across nets."""
    total = 0.0
    count = 0
    for i in range(n):
        if len(candidates[i]) <= 1:
            continue
        wls = [c.wirelength for c in candidates[i]]
        vias = [c.num_vias for c in candidates[i]]
        total += alpha * (max(wls) - min(wls)) + gamma * (max(vias) - min(vias))
        count += 1
    return max(total / max(count, 1), 1.0)


def _print_stats(label, routes, nets, params, max_coupling_distance=None):
    wl = sum(r.wirelength for r in routes)
    xt = total_crosstalk(routes, params, max_coupling_distance)
    nv = sum(r.num_vias for r in routes)
    xings = total_crossings(routes)
    drc = total_drc_violations(routes, nets, params)
    print(f"  [{label}] WL={wl:.1f}  XT={xt:.2f}  "
          f"Vias={nv}  Xings={xings}  DRC={drc}")


def generate_all_candidates(package: Package, verbose: bool = False):
    """Generate routing candidates for every net."""
    candidates = []
    for net in package.nets:
        cands = generate_candidates(
            net_idx=net.idx, pt1=net.pt1, pt2=net.pt2,
            num_layers=package.params.num_layers,
        )
        candidates.append(cands)
        if verbose:
            print(f"  {net.name}: {len(cands)} candidates")
    return candidates
