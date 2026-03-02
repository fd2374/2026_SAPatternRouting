"""
Route selection solver: SA with adaptive reheating.

Geometric cooling. Reheat is triggered when the energy moving average
stagnates (relative change below threshold over REHEAT_WINDOW iterations).
Temperature target halves with each reheat.

Single-net perturbation per step gives O(n) incremental delta.
"""

import math
import random
from collections import deque
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

COOL_RATE = 0.98
REHEAT_WINDOW = 200          # moving-average window for stagnation detection
STAGNATION_THRESHOLD = 1e-3  # relative change below which we reheat


def solve_sa(
    package: Package,
    candidates: List[List[RoutePattern]],
    alpha: float = 1.0,
    beta: float = 10.0,
    gamma: float = 5.0,
    max_iters: int = 5000,
    max_coupling_distance: float = None,
    verbose: bool = True,
    live_plot: bool = False,
) -> RoutingSolution:
    """SA solver with geometric cooling and adaptive reheating."""
    params = package.params
    nets = package.nets
    n = len(nets)

    if max_coupling_distance is None:
        max_coupling_distance = compute_max_coupling_distance(params)

    if n == 0:
        return RoutingSolution(package=package, routes=[])

    # --- Pre-extract numpy arrays ---
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

    def _full_energy():
        wl = sum(r.wirelength for r in routes)
        nv = sum(r.num_vias for r in routes)
        pair = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                pair += pt(cur_seg[i], cur_seg[j], cur_obs[i], cur_obs[j])
        return alpha * wl + gamma * nv + pair

    energy = _full_energy()
    best_energy = energy
    best_selected = list(selected)

    # --- Temperature: set T0 so initial acceptance rate ≈ 80% ---
    T0 = _init_temperature(n, candidates, selected, routes,
                           seg_arrs, obs_arrs, cur_seg, cur_obs, pt,
                           alpha, gamma)
    temperature = T0
    reheat_temp = T0
    steps_per_iter = max(20, n * 3)
    energy_history: deque = deque(maxlen=REHEAT_WINDOW)
    n_reheats = 0

    if verbose:
        print(f"[SA] {n} nets, T0={T0:.1f}, cool={COOL_RATE}, "
              f"reheat_window={REHEAT_WINDOW}, "
              f"stagnation_thr={STAGNATION_THRESHOLD}")
        print(f"[SA] {steps_per_iter} steps/iter, {max_iters} iters")
        _print_stats("init", routes, nets, params, max_coupling_distance)
        print(f"  E={energy:.1f}")

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

            delta = alpha * (new_route.wirelength - old_route.wirelength)
            delta += gamma * (new_route.num_vias - old_route.num_vias)
            for j in range(n):
                if j == i:
                    continue
                delta += (pt(new_sa, cur_seg[j], new_oa, cur_obs[j]) -
                          pt(cur_seg[i], cur_seg[j], cur_obs[i], cur_obs[j]))

            if delta <= 0 or random.random() < math.exp(
                    -delta / max(temperature, 1e-30)):
                selected[i] = new_k
                routes[i] = new_route
                cur_seg[i] = new_sa
                cur_obs[i] = new_oa
                energy += delta
                accepted += 1
                if energy < best_energy:
                    best_energy = energy
                    best_selected = list(selected)

        temperature *= COOL_RATE

        # Stagnation detection via moving average
        energy_history.append(energy)
        if len(energy_history) == REHEAT_WINDOW:
            half = REHEAT_WINDOW // 2
            hist = list(energy_history)
            avg_old = sum(hist[:half]) / half
            avg_new = sum(hist[half:]) / half
            rel_change = abs(avg_new - avg_old) / max(abs(avg_old), 1e-30)
            if rel_change < STAGNATION_THRESHOLD:
                reheat_temp *= 0.5
                temperature = reheat_temp
                energy_history.clear()
                n_reheats += 1
                # Restart from best solution
                selected = list(best_selected)
                routes = [candidates[i][selected[i]] for i in range(n)]
                cur_seg = [seg_arrs[i][selected[i]] for i in range(n)]
                cur_obs = [obs_arrs[i][selected[i]] for i in range(n)]
                energy = _full_energy()
                if sa_plot is not None:
                    sa_plot.mark_reheat(t + 1)
                if verbose:
                    print(f"  [reheat #{n_reheats}] T -> {temperature:.1f}"
                          f"  (restart from E*={best_energy:.1f})")

        acc_rate = accepted / steps_per_iter
        if sa_plot is not None and (t < 100 or t % 10 == 0):
            sa_plot.update(t + 1, energy, best_energy,
                           temperature, acc_rate)

        if verbose and (t < 3 or (t + 1) % 200 == 0 or t == max_iters - 1):
            _print_stats(f"iter {t+1}", routes, nets, params,
                         max_coupling_distance)
            print(f"  T={temperature:.4g}  acc={accepted}/{steps_per_iter}  "
                  f"E={energy:.1f}  E*={best_energy:.1f}  "
                  f"reheats={n_reheats}")

    if sa_plot is not None:
        sa_plot.finish()

    # Restore best
    selected = best_selected
    routes = [candidates[i][selected[i]] for i in range(n)]

    if verbose:
        _print_stats("best", routes, nets, params, max_coupling_distance)
        print(f"  reheats={n_reheats}")

    return RoutingSolution(
        package=package,
        routes=routes,
        total_wirelength=sum(r.wirelength for r in routes),
        total_crosstalk=total_crosstalk(routes, params, max_coupling_distance),
        max_crosstalk=max_pair_crosstalk(routes, params, max_coupling_distance),
        num_vias=sum(r.num_vias for r in routes),
        num_crossings=total_crossings(routes),
        num_drc_violations=total_drc_violations(routes, nets, params),
    )


def _init_temperature(n, candidates, selected, routes,
                      seg_arrs, obs_arrs, cur_seg, cur_obs, pt,
                      alpha, gamma, target_acc=0.8):
    """Estimate T0 by sampling random perturbations so that
    the initial acceptance rate ≈ target_acc (default 80%).
    Uses T0 = -Δ_avg / ln(target_acc)."""
    n_samples = max(500, n * 20)
    uphill = []
    for _ in range(n_samples):
        i = random.randrange(n)
        if len(candidates[i]) <= 1:
            continue
        new_k = random.randrange(len(candidates[i]))
        if new_k == selected[i]:
            continue
        new_r = candidates[i][new_k]
        old_r = routes[i]
        d = alpha * (new_r.wirelength - old_r.wirelength)
        d += gamma * (new_r.num_vias - old_r.num_vias)
        for j in range(n):
            if j == i:
                continue
            d += (pt(seg_arrs[i][new_k], cur_seg[j],
                     obs_arrs[i][new_k], cur_obs[j]) -
                  pt(cur_seg[i], cur_seg[j], cur_obs[i], cur_obs[j]))
        if d > 0:
            uphill.append(d)
    if not uphill:
        return 1.0
    delta_avg = sum(uphill) / len(uphill)
    return delta_avg / math.log(1.0 / target_acc)


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
