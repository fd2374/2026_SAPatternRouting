"""
Post-processing: greedy violation removal + A* rerouting.

Step 1 – Greedy removal:
  Count per-net violations (crossings + DRC) against all other nets.
  Repeatedly remove the net with the most violations until none remain.

Step 2 – A* rerouting:
  Re-route removed nets on a multi-layer grid with 135-degree constraint,
  treating all remaining wires/vias as obstacles.
"""

import math
import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional, Set

from .data_types import (
    Point, Segment, Via, RoutePattern, Net, DesignParams, Package, RoutingSolution,
)
from .util import (
    route_pair_crossings, route_pair_drc, get_via_obstacles,
    total_crossings, total_drc_violations,
    total_crosstalk, max_pair_crosstalk,
)

GRID_STEP = 1.0
_SQRT2 = math.sqrt(2.0)


def _count_violations_per_net(
    routes: List[RoutePattern],
    nets: List[Net],
    params: DesignParams,
) -> List[int]:
    """Count total violations (crossings + DRC) for each net vs all others."""
    n = len(routes)
    counts = [0] * n
    all_obs = [get_via_obstacles(routes[i], nets[i], params) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            xings = route_pair_crossings(routes[i], routes[j])
            drc = route_pair_drc(routes[i], routes[j],
                                 all_obs[i], all_obs[j], params)
            v = xings + drc
            counts[i] += v
            counts[j] += v
    return counts


def greedy_remove(
    solution: RoutingSolution,
    verbose: bool = True,
) -> Tuple[RoutingSolution, List[int]]:
    """
    Greedily remove nets with violations until the solution is clean.

    Returns:
        updated_solution: RoutingSolution with violating nets removed
                          (removed nets get an empty placeholder route)
        removed_indices:  list of net indices that were removed, in removal order
    """
    package = solution.package
    params = package.params
    nets = package.nets
    n = len(nets)

    routes = list(solution.routes)
    removed: List[int] = []
    alive = set(range(n))

    iteration = 0
    while True:
        alive_list = sorted(alive)
        alive_routes = [routes[i] for i in alive_list]
        alive_nets = [nets[i] for i in alive_list]

        counts_alive = _count_violations_per_net(alive_routes, alive_nets, params)

        total_v = sum(counts_alive) // 2
        if total_v == 0:
            break

        worst_local = max(range(len(alive_list)), key=lambda k: counts_alive[k])
        worst_global = alive_list[worst_local]

        iteration += 1
        if verbose:
            print(f"  [remove #{iteration}] net{worst_global} "
                  f"({nets[worst_global].name}) has {counts_alive[worst_local]} "
                  f"violations (total remaining: {total_v})")

        alive.discard(worst_global)
        removed.append(worst_global)

    if verbose:
        print(f"  [greedy] Removed {len(removed)} nets, "
              f"{len(alive)} nets remain violation-free")

    alive_routes = [routes[i] for i in sorted(alive)]

    updated = RoutingSolution(
        package=package,
        routes=routes,
        total_wirelength=sum(r.wirelength for r in alive_routes),
        total_crosstalk=total_crosstalk(alive_routes, params),
        max_crosstalk=max_pair_crosstalk(alive_routes, params),
        num_vias=sum(r.num_vias for r in alive_routes),
        num_crossings=total_crossings(alive_routes),
        num_drc_violations=total_drc_violations(
            alive_routes, [nets[i] for i in sorted(alive)], params),
    )

    return updated, removed


# ============================================================
# Step 2: A* rerouting
# ============================================================

def _pt_seg_dist(px, py, x1, y1, x2, y2):
    """Point-to-segment distance."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-18:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _mark_segment(blocked, seg_sx, seg_sy, seg_ex, seg_ey, layer,
                  clearance, grid_step, nx, ny):
    """Mark grid cells near a wire segment as blocked."""
    gx_lo = max(0, int(math.floor((min(seg_sx, seg_ex) - clearance) / grid_step)))
    gx_hi = min(nx - 1, int(math.ceil((max(seg_sx, seg_ex) + clearance) / grid_step)))
    gy_lo = max(0, int(math.floor((min(seg_sy, seg_ey) - clearance) / grid_step)))
    gy_hi = min(ny - 1, int(math.ceil((max(seg_sy, seg_ey) + clearance) / grid_step)))
    for gy in range(gy_lo, gy_hi + 1):
        py = gy * grid_step
        for gx in range(gx_lo, gx_hi + 1):
            px = gx * grid_step
            if _pt_seg_dist(px, py, seg_sx, seg_sy, seg_ex, seg_ey) < clearance:
                blocked[layer, gy, gx] = True


def _mark_via(blocked, cx, cy, clearance, layers, grid_step, nx, ny):
    """Mark grid cells near a via as blocked (square footprint)."""
    gx_lo = max(0, int(math.floor((cx - clearance) / grid_step)))
    gx_hi = min(nx - 1, int(math.ceil((cx + clearance) / grid_step)))
    gy_lo = max(0, int(math.floor((cy - clearance) / grid_step)))
    gy_hi = min(ny - 1, int(math.ceil((cy + clearance) / grid_step)))
    for gy in range(gy_lo, gy_hi + 1):
        py = gy * grid_step
        for gx in range(gx_lo, gx_hi + 1):
            px = gx * grid_step
            if abs(px - cx) < clearance and abs(py - cy) < clearance:
                for layer in layers:
                    blocked[layer, gy, gx] = True


def _clearances(params, grid_step):
    """Compute wire and via clearances for obstacle marking.

    The clearance must be at least grid_step * sqrt(2)/2 so that the
    blocked band forms a continuous wall on the grid — preventing A*
    diagonal moves from hopping over existing wires (crossing prevention).
    """
    min_wall = grid_step * _SQRT2 * 0.5 + 1.0
    wire_clr = max(params.min_center_distance, min_wall)
    via_clr = max((params.wire_width + params.via_width) / 2.0, min_wall)
    return wire_clr, via_clr


def _build_obstacle_grid(routes, nets, params, grid_step, exclude):
    """Build a 3D boolean grid [layer, y, x] marking blocked cells."""
    pkg_w, pkg_h = params.pkg_width, params.pkg_height
    nl = params.num_layers
    nx = int(math.ceil(pkg_w / grid_step)) + 1
    ny = int(math.ceil(pkg_h / grid_step)) + 1
    blocked = np.zeros((nl, ny, nx), dtype=np.bool_)

    wire_clr, via_clr = _clearances(params, grid_step)

    for idx, route in enumerate(routes):
        if idx in exclude or route is None:
            continue
        for seg in route.segments:
            _mark_segment(blocked, seg.start.x, seg.start.y,
                          seg.end.x, seg.end.y, seg.layer,
                          wire_clr, grid_step, nx, ny)
        obs = get_via_obstacles(route, nets[idx], params)
        for vob in obs:
            _mark_via(blocked, vob.center.x, vob.center.y,
                      via_clr, list(vob.layers), grid_step, nx, ny)

    return blocked, nx, ny


def _add_route_to_grid(blocked, route, net, params, grid_step, nx, ny):
    """Add a single route's obstacles to the grid."""
    wire_clr, via_clr = _clearances(params, grid_step)
    for seg in route.segments:
        _mark_segment(blocked, seg.start.x, seg.start.y,
                      seg.end.x, seg.end.y, seg.layer,
                      wire_clr, grid_step, nx, ny)
    obs = get_via_obstacles(route, net, params)
    for vob in obs:
        _mark_via(blocked, vob.center.x, vob.center.y,
                  via_clr, list(vob.layers), grid_step, nx, ny)


def _astar(blocked, sx, sy, ex, ey, nx, ny, num_layers, grid_step):
    """
    A* on a multi-layer grid with 135-degree moves.

    Start/goal: all layers at (sx,sy) / (ex,ey).
    Returns list of (gx, gy, layer) or None.
    """
    VIA_COST = grid_step * 2.0
    DIRS = [
        (1, 0, 1.0), (-1, 0, 1.0),
        (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, _SQRT2), (-1, -1, _SQRT2),
        (1, -1, _SQRT2), (-1, 1, _SQRT2),
    ]

    def heuristic(x, y):
        dx = abs(x - ex)
        dy = abs(y - ey)
        return (max(dx, dy) + (_SQRT2 - 1.0) * min(dx, dy)) * grid_step

    # Push all layers at start
    open_set = []
    g_score: Dict[Tuple[int, int, int], float] = {}
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    goals = set()

    for layer in range(num_layers):
        state = (sx, sy, layer)
        g_score[state] = 0.0
        heapq.heappush(open_set, (heuristic(sx, sy), 0.0, sx, sy, layer))
        goals.add((ex, ey, layer))

    while open_set:
        f, g, cx, cy, cl = heapq.heappop(open_set)

        current = (cx, cy, cl)
        if current in goals:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if g > g_score.get(current, float('inf')):
            continue

        # Same-layer moves
        for dx, dy, cost_m in DIRS:
            nx2 = cx + dx
            ny2 = cy + dy
            if 0 <= nx2 < nx and 0 <= ny2 < ny:
                if not blocked[cl, ny2, nx2]:
                    new_g = g + cost_m * grid_step
                    nb = (nx2, ny2, cl)
                    if new_g < g_score.get(nb, float('inf')):
                        g_score[nb] = new_g
                        came_from[nb] = (cx, cy, cl)
                        heapq.heappush(open_set,
                                       (new_g + heuristic(nx2, ny2),
                                        new_g, nx2, ny2, cl))

        # Layer transitions
        for dl in (-1, 1):
            nl = cl + dl
            if 0 <= nl < num_layers and not blocked[nl, cy, cx]:
                new_g = g + VIA_COST
                nb = (cx, cy, nl)
                if new_g < g_score.get(nb, float('inf')):
                    g_score[nb] = new_g
                    came_from[nb] = (cx, cy, cl)
                    heapq.heappush(open_set,
                                   (new_g + heuristic(cx, cy),
                                    new_g, cx, cy, nl))

    return None


def _path_to_route(path, grid_step, net_idx, net):
    """Convert a grid path [(gx,gy,layer),...] into a RoutePattern."""
    if len(path) < 2:
        return None

    # Split path into same-layer runs; vias at layer transitions
    runs = []
    vias = []
    cur_layer = path[0][2]
    cur_run = [(path[0][0], path[0][1])]

    for i in range(1, len(path)):
        gx, gy, layer = path[i]
        if layer == cur_layer:
            cur_run.append((gx, gy))
        else:
            runs.append((cur_layer, cur_run))
            prev = cur_run[-1]
            vias.append(Via(
                pos=Point(prev[0] * grid_step, prev[1] * grid_step),
                from_layer=cur_layer, to_layer=layer))
            cur_layer = layer
            cur_run = [(gx, gy)]
    runs.append((cur_layer, cur_run))

    # Merge collinear consecutive points into segments
    segments = []
    for layer, pts in runs:
        if len(pts) < 2:
            continue
        seg_start = 0
        for i in range(2, len(pts)):
            dx_prev = pts[i - 1][0] - pts[seg_start][0]
            dy_prev = pts[i - 1][1] - pts[seg_start][1]
            dx_cur = pts[i][0] - pts[seg_start][0]
            dy_cur = pts[i][1] - pts[seg_start][1]
            if dx_prev * dy_cur != dy_prev * dx_cur:
                segments.append(Segment(
                    start=Point(pts[seg_start][0] * grid_step,
                                pts[seg_start][1] * grid_step),
                    end=Point(pts[i - 1][0] * grid_step,
                              pts[i - 1][1] * grid_step),
                    layer=layer))
                seg_start = i - 1
        segments.append(Segment(
            start=Point(pts[seg_start][0] * grid_step,
                        pts[seg_start][1] * grid_step),
            end=Point(pts[-1][0] * grid_step, pts[-1][1] * grid_step),
            layer=layer))

    # Connect to actual pad positions if grid-snapped points differ
    if segments:
        p1 = net.pt1
        first_start = segments[0].start
        if p1.distance_to(first_start) > 1e-6:
            segments.insert(0, Segment(start=p1, end=first_start,
                                       layer=segments[0].layer))
        p2 = net.pt2
        last_end = segments[-1].end
        if p2.distance_to(last_end) > 1e-6:
            segments.append(Segment(start=last_end, end=p2,
                                    layer=segments[-1].layer))

    return RoutePattern(net_idx=net_idx, pattern_idx=-1,
                        segments=segments, vias=vias)


def astar_reroute(solution, removed_indices, grid_step=GRID_STEP,
                  verbose=True):
    """Reroute removed nets via A* and return a complete RoutingSolution."""
    package = solution.package
    params = package.params
    nets = package.nets
    n = len(nets)
    num_layers = params.num_layers

    all_routes = list(solution.routes)
    exclude = set(removed_indices)

    if verbose:
        print(f"  [A*] Building obstacle grid (step={grid_step})...")
    blocked, nx, ny = _build_obstacle_grid(
        all_routes, nets, params, grid_step, exclude)
    if verbose:
        print(f"  [A*] Grid: {nx}x{ny}x{num_layers}")

    rerouted_routes = list(all_routes)

    for net_idx in removed_indices:
        net = nets[net_idx]
        sx = max(0, min(nx - 1, round(net.pt1.x / grid_step)))
        sy = max(0, min(ny - 1, round(net.pt1.y / grid_step)))
        ex_g = max(0, min(nx - 1, round(net.pt2.x / grid_step)))
        ey_g = max(0, min(ny - 1, round(net.pt2.y / grid_step)))

        # Temporarily unblock start/goal on all layers
        s_bak = blocked[:, sy, sx].copy()
        g_bak = blocked[:, ey_g, ex_g].copy()
        blocked[:, sy, sx] = False
        blocked[:, ey_g, ex_g] = False

        path = _astar(blocked, sx, sy, ex_g, ey_g,
                       nx, ny, num_layers, grid_step)

        blocked[:, sy, sx] = s_bak
        blocked[:, ey_g, ex_g] = g_bak

        if path is None:
            if verbose:
                print(f"  [A*] net{net_idx} ({net.name}): NO PATH FOUND")
            continue

        route = _path_to_route(path, grid_step, net_idx, net)
        if route is None:
            continue

        rerouted_routes[net_idx] = route
        _add_route_to_grid(blocked, route, net, params, grid_step, nx, ny)

        if verbose:
            print(f"  [A*] net{net_idx} ({net.name}): "
                  f"WL={route.wirelength:.1f} vias={route.num_vias} "
                  f"segs={len(route.segments)}")

    final = RoutingSolution(
        package=package,
        routes=rerouted_routes,
        total_wirelength=sum(r.wirelength for r in rerouted_routes),
        total_crosstalk=total_crosstalk(rerouted_routes, params),
        max_crosstalk=max_pair_crosstalk(rerouted_routes, params),
        num_vias=sum(r.num_vias for r in rerouted_routes),
        num_crossings=total_crossings(rerouted_routes),
        num_drc_violations=total_drc_violations(rerouted_routes, nets, params),
    )
    return final
