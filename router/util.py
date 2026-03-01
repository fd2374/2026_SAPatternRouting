"""
Routing utility functions: crosstalk, crossing detection, DRC checks.

Crosstalk model:
  coupling(seg_i, seg_j) = coupling_length / spacing
  Only between parallel segments on the SAME layer.

DRC model (design rule checks):
  - Wire-via collision: wire (with WIREWIDTH) must not overlap via (VIAWIDTH square)
  - Via-via collision: two vias must not overlap on shared occupied layers
  - IO pads are modeled as special vias occupying layers 0..connection_layer
  - Via occupies all layers between from_layer and to_layer
"""

import math
import numpy as np
from numba import njit
from typing import List, Tuple

from .data_types import (
    Point, Segment, RoutePattern, Net, Direction, DesignParams, ViaObstacle,
)

SQRT2 = math.sqrt(2.0)
_SQRT2 = np.float64(SQRT2)


def get_via_obstacles(
    route: RoutePattern, net: Net, params: DesignParams,
) -> List[ViaObstacle]:
    """
    Extract all physical via obstacles from a route, including pad-vias.

    - Regular vias: occupy layers from_layer..to_layer, size = via_width.
    - Pad vias: IO pad at each net endpoint, size = via_width,
      occupies layers 0..connection_layer (pad is above routing layers,
      so a via stack punches down from the chip to the routing layer).
    """
    via_w = params.via_width
    obstacles: List[ViaObstacle] = []

    for via in route.vias:
        lo = min(via.from_layer, via.to_layer)
        hi = max(via.from_layer, via.to_layer)
        obstacles.append(ViaObstacle(via.pos, via_w, frozenset(range(lo, hi + 1))))

    if route.segments:
        pad1_layer = route.segments[0].layer
        pad1_layers = frozenset(range(0, pad1_layer + 1))
        obstacles.append(ViaObstacle(net.pt1, via_w, pad1_layers))

        pad2_layer = route.segments[-1].layer
        pad2_layers = frozenset(range(0, pad2_layer + 1))
        obstacles.append(ViaObstacle(net.pt2, via_w, pad2_layers))

    return obstacles


# ============================================================
# DRC: geometry helpers
# ============================================================

def _point_seg_dist(px: float, py: float,
                    x1: float, y1: float,
                    x2: float, y2: float) -> float:
    """Minimum distance from point (px, py) to segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-18:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _wire_via_overlap(seg: Segment, via: ViaObstacle,
                      wire_width: float) -> bool:
    """Check if a wire segment (with width) overlaps a via on a shared layer."""
    if seg.layer not in via.layers:
        return False
    dist = _point_seg_dist(via.center.x, via.center.y,
                           seg.start.x, seg.start.y,
                           seg.end.x, seg.end.y)
    return dist < (wire_width + via.width) / 2.0 - 1e-9


def _via_via_overlap(a: ViaObstacle, b: ViaObstacle) -> bool:
    """Check if two square vias overlap on any shared layer."""
    if a.layers.isdisjoint(b.layers):
        return False
    half = (a.width + b.width) / 2.0
    return (abs(a.center.x - b.center.x) < half - 1e-9 and
            abs(a.center.y - b.center.y) < half - 1e-9)


# ============================================================
# DRC: pairwise route check
# ============================================================

def route_pair_drc(
    route_a: RoutePattern, route_b: RoutePattern,
    via_obs_a: List[ViaObstacle], via_obs_b: List[ViaObstacle],
    params: DesignParams,
) -> int:
    """
    Count DRC violations between two routes.

    Checks:
      1. route_a wires vs route_b via obstacles
      2. route_b wires vs route_a via obstacles
      3. route_a via obstacles vs route_b via obstacles
    """
    wire_w = params.wire_width
    count = 0

    for seg in route_a.segments:
        for vob in via_obs_b:
            if _wire_via_overlap(seg, vob, wire_w):
                count += 1

    for seg in route_b.segments:
        for vob in via_obs_a:
            if _wire_via_overlap(seg, vob, wire_w):
                count += 1

    for va in via_obs_a:
        for vb in via_obs_b:
            if _via_via_overlap(va, vb):
                count += 1

    return count


def total_drc_violations(
    routes: List[RoutePattern],
    nets: List[Net],
    params: DesignParams,
) -> int:
    """Count total DRC violations across all route pairs."""
    n = len(routes)
    all_obs = [get_via_obstacles(routes[i], nets[i], params) for i in range(n)]
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += route_pair_drc(routes[i], routes[j],
                                    all_obs[i], all_obs[j], params)
    return total


# ============================================================
# Crosstalk: parallel coupling model
# ============================================================

def segment_coupling(
    s1: Segment, s2: Segment, min_spacing: float, max_distance: float,
) -> float:
    """
    Crosstalk coupling cost between two parallel same-layer segments.
    Returns coupling_length / spacing, or 0 if not parallel / too far.
    """
    if s1.layer != s2.layer:
        return 0.0
    d1 = s1.direction
    d2 = s2.direction
    if d1 != d2:
        return 0.0
    if d1 == Direction.HORIZONTAL:
        return _coupling_horizontal(s1, s2, min_spacing, max_distance)
    elif d1 == Direction.VERTICAL:
        return _coupling_vertical(s1, s2, min_spacing, max_distance)
    elif d1 == Direction.DIAG_POS:
        return _coupling_diag_pos(s1, s2, min_spacing, max_distance)
    return _coupling_diag_neg(s1, s2, min_spacing, max_distance)


def _coupling_horizontal(s1, s2, min_spacing, max_dist):
    y1 = (s1.start.y + s1.end.y) / 2.0
    y2 = (s2.start.y + s2.end.y) / 2.0
    spacing = abs(y1 - y2)
    if spacing < 1e-9 or spacing > max_dist:
        return 0.0
    overlap = max(0.0, min(s1.max_x, s2.max_x) - max(s1.min_x, s2.min_x))
    if overlap < 1e-9:
        return 0.0
    return overlap / max(spacing, min_spacing)


def _coupling_vertical(s1, s2, min_spacing, max_dist):
    x1 = (s1.start.x + s1.end.x) / 2.0
    x2 = (s2.start.x + s2.end.x) / 2.0
    spacing = abs(x1 - x2)
    if spacing < 1e-9 or spacing > max_dist:
        return 0.0
    overlap = max(0.0, min(s1.max_y, s2.max_y) - max(s1.min_y, s2.min_y))
    if overlap < 1e-9:
        return 0.0
    return overlap / max(spacing, min_spacing)


def _coupling_diag_pos(s1, s2, min_spacing, max_dist):
    perp = abs(
        (s2.start.x - s1.start.x) - (s2.start.y - s1.start.y)
    ) / SQRT2
    if perp < 1e-9 or perp > max_dist:
        return 0.0

    def proj(p):
        return (p.x + p.y) / SQRT2

    lo1, hi1 = min(proj(s1.start), proj(s1.end)), max(proj(s1.start), proj(s1.end))
    lo2, hi2 = min(proj(s2.start), proj(s2.end)), max(proj(s2.start), proj(s2.end))
    overlap = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    if overlap < 1e-9:
        return 0.0
    return overlap / max(perp, min_spacing)


def _coupling_diag_neg(s1, s2, min_spacing, max_dist):
    perp = abs(
        (s2.start.x - s1.start.x) + (s2.start.y - s1.start.y)
    ) / SQRT2
    if perp < 1e-9 or perp > max_dist:
        return 0.0

    def proj(p):
        return (p.x - p.y) / SQRT2

    lo1, hi1 = min(proj(s1.start), proj(s1.end)), max(proj(s1.start), proj(s1.end))
    lo2, hi2 = min(proj(s2.start), proj(s2.end)), max(proj(s2.start), proj(s2.end))
    overlap = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    if overlap < 1e-9:
        return 0.0
    return overlap / max(perp, min_spacing)


# ============================================================
# Same-layer crossing detection
# ============================================================

def _cross2d(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _on_segment(px, py, qx, qy, rx, ry):
    return (min(px, rx) <= qx + 1e-9 <= max(px, rx) + 1e-9 and
            min(py, ry) <= qy + 1e-9 <= max(py, ry) + 1e-9)


def segments_intersect(s1: Segment, s2: Segment) -> bool:
    """Check if two segments on the same layer properly intersect."""
    if s1.layer != s2.layer:
        return False
    ax, ay = s1.start.x, s1.start.y
    bx, by = s1.end.x, s1.end.y
    cx, cy = s2.start.x, s2.start.y
    dx, dy = s2.end.x, s2.end.y

    d1 = _cross2d(cx, cy, dx, dy, ax, ay)
    d2 = _cross2d(cx, cy, dx, dy, bx, by)
    d3 = _cross2d(ax, ay, bx, by, cx, cy)
    d4 = _cross2d(ax, ay, bx, by, dx, dy)

    if ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)):
        if ((d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)):
            return True

    if abs(d1) < 1e-9 and _on_segment(cx, cy, ax, ay, dx, dy):
        return True
    if abs(d2) < 1e-9 and _on_segment(cx, cy, bx, by, dx, dy):
        return True
    if abs(d3) < 1e-9 and _on_segment(ax, ay, cx, cy, bx, by):
        return True
    if abs(d4) < 1e-9 and _on_segment(ax, ay, dx, dy, bx, by):
        return True
    return False


def _bbox_overlap(a, b, margin: float = 0.0) -> bool:
    return not (a[2] + margin < b[0] or b[2] + margin < a[0] or
                a[3] + margin < b[1] or b[3] + margin < a[1])


def route_pair_crossings(route_a: RoutePattern, route_b: RoutePattern) -> int:
    """Count same-layer segment crossings between two routes."""
    if route_a.layers_used.isdisjoint(route_b.layers_used):
        return 0
    if not _bbox_overlap(route_a.bbox, route_b.bbox):
        return 0
    count = 0
    for sa in route_a.segments:
        for sb in route_b.segments:
            if segments_intersect(sa, sb):
                count += 1
    return count


def total_crossings(routes: List[RoutePattern]) -> int:
    total = 0
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            total += route_pair_crossings(routes[i], routes[j])
    return total


# ============================================================
# Crosstalk aggregation
# ============================================================

def route_pair_crosstalk(
    route_a: RoutePattern, route_b: RoutePattern, params: DesignParams,
    max_coupling_distance: float = None,
) -> float:
    """Total crosstalk cost between two route patterns."""
    if route_a.layers_used.isdisjoint(route_b.layers_used):
        return 0.0
    min_spacing = params.min_center_distance
    if max_coupling_distance is None:
        pkg_diag = math.hypot(params.pkg_width, params.pkg_height)
        max_coupling_distance = max(100 * min_spacing, pkg_diag * 0.1)
    if not _bbox_overlap(route_a.bbox, route_b.bbox, max_coupling_distance):
        return 0.0
    total = 0.0
    for sa in route_a.segments:
        for sb in route_b.segments:
            total += segment_coupling(sa, sb, min_spacing, max_coupling_distance)
    return total


def compute_max_coupling_distance(params: DesignParams) -> float:
    pkg_diag = math.hypot(params.pkg_width, params.pkg_height)
    return max(100 * params.min_center_distance, pkg_diag * 0.1)


def total_crosstalk(
    routes: List[RoutePattern], params: DesignParams,
    max_coupling_distance: float = None,
) -> float:
    total = 0.0
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            total += route_pair_crosstalk(routes[i], routes[j], params,
                                          max_coupling_distance)
    return total


def max_pair_crosstalk(
    routes: List[RoutePattern], params: DesignParams,
    max_coupling_distance: float = None,
) -> float:
    worst = 0.0
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            xt = route_pair_crosstalk(routes[i], routes[j], params,
                                      max_coupling_distance)
            worst = max(worst, xt)
    return worst


# ============================================================
# Numba-accelerated core (used by solver SA inner loop)
# ============================================================
# Segment array:      shape (n, 5) = [start_x, start_y, end_x, end_y, layer]
# Via obstacle array:  shape (n, 5) = [center_x, center_y, width, min_layer, max_layer]

_EMPTY_SEG = np.empty((0, 5), dtype=np.float64)
_EMPTY_OBS = np.empty((0, 5), dtype=np.float64)

_DIR_MAP = {Direction.HORIZONTAL: 0, Direction.VERTICAL: 1,
            Direction.DIAG_POS: 2, Direction.DIAG_NEG: 3}


def extract_seg_array(route: RoutePattern) -> np.ndarray:
    n = len(route.segments)
    if n == 0:
        return _EMPTY_SEG
    arr = np.empty((n, 5), dtype=np.float64)
    for i, s in enumerate(route.segments):
        arr[i, 0] = s.start.x
        arr[i, 1] = s.start.y
        arr[i, 2] = s.end.x
        arr[i, 3] = s.end.y
        arr[i, 4] = s.layer
    return arr


def extract_obs_array(obs_list: List[ViaObstacle]) -> np.ndarray:
    n = len(obs_list)
    if n == 0:
        return _EMPTY_OBS
    arr = np.empty((n, 5), dtype=np.float64)
    for i, ob in enumerate(obs_list):
        arr[i, 0] = ob.center.x
        arr[i, 1] = ob.center.y
        arr[i, 2] = ob.width
        arr[i, 3] = min(ob.layers)
        arr[i, 4] = max(ob.layers)
    return arr


@njit(cache=True)
def _get_dir(sx, sy, ex, ey):
    dx = ex - sx
    dy = ey - sy
    if abs(dy) < 1e-9:
        return 0
    if abs(dx) < 1e-9:
        return 1
    if dx * dy > 0:
        return 2
    return 3


@njit(cache=True)
def _seg_intersect_jit(ax, ay, bx, by, cx, cy, dx, dy):
    d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
    d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)

    if ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)):
        if ((d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)):
            return True

    if abs(d1) < 1e-9:
        if (min(cx, dx) <= ax + 1e-9 <= max(cx, dx) + 1e-9 and
                min(cy, dy) <= ay + 1e-9 <= max(cy, dy) + 1e-9):
            return True
    if abs(d2) < 1e-9:
        if (min(cx, dx) <= bx + 1e-9 <= max(cx, dx) + 1e-9 and
                min(cy, dy) <= by + 1e-9 <= max(cy, dy) + 1e-9):
            return True
    if abs(d3) < 1e-9:
        if (min(ax, bx) <= cx + 1e-9 <= max(ax, bx) + 1e-9 and
                min(ay, by) <= cy + 1e-9 <= max(ay, by) + 1e-9):
            return True
    if abs(d4) < 1e-9:
        if (min(ax, bx) <= dx + 1e-9 <= max(ax, bx) + 1e-9 and
                min(ay, by) <= dy + 1e-9 <= max(ay, by) + 1e-9):
            return True
    return False


@njit(cache=True)
def _coupling_jit(s1_sx, s1_sy, s1_ex, s1_ey,
                  s2_sx, s2_sy, s2_ex, s2_ey,
                  direction, min_spacing, max_dist):
    SQRT2 = 1.4142135623730951
    if direction == 0:  # HORIZONTAL
        y1 = (s1_sy + s1_ey) * 0.5
        y2 = (s2_sy + s2_ey) * 0.5
        spacing = abs(y1 - y2)
        if spacing < 1e-9 or spacing > max_dist:
            return 0.0
        overlap = min(max(s1_sx, s1_ex), max(s2_sx, s2_ex)) - max(min(s1_sx, s1_ex), min(s2_sx, s2_ex))
        if overlap < 1e-9:
            return 0.0
        return overlap / max(spacing, min_spacing)
    elif direction == 1:  # VERTICAL
        x1 = (s1_sx + s1_ex) * 0.5
        x2 = (s2_sx + s2_ex) * 0.5
        spacing = abs(x1 - x2)
        if spacing < 1e-9 or spacing > max_dist:
            return 0.0
        overlap = min(max(s1_sy, s1_ey), max(s2_sy, s2_ey)) - max(min(s1_sy, s1_ey), min(s2_sy, s2_ey))
        if overlap < 1e-9:
            return 0.0
        return overlap / max(spacing, min_spacing)
    elif direction == 2:  # DIAG_POS
        perp = abs((s2_sx - s1_sx) - (s2_sy - s1_sy)) / SQRT2
        if perp < 1e-9 or perp > max_dist:
            return 0.0
        lo1 = min(s1_sx + s1_sy, s1_ex + s1_ey) / SQRT2
        hi1 = max(s1_sx + s1_sy, s1_ex + s1_ey) / SQRT2
        lo2 = min(s2_sx + s2_sy, s2_ex + s2_ey) / SQRT2
        hi2 = max(s2_sx + s2_sy, s2_ex + s2_ey) / SQRT2
        overlap = min(hi1, hi2) - max(lo1, lo2)
        if overlap < 1e-9:
            return 0.0
        return overlap / max(perp, min_spacing)
    else:  # DIAG_NEG
        perp = abs((s2_sx - s1_sx) + (s2_sy - s1_sy)) / SQRT2
        if perp < 1e-9 or perp > max_dist:
            return 0.0
        lo1 = min(s1_sx - s1_sy, s1_ex - s1_ey) / SQRT2
        hi1 = max(s1_sx - s1_sy, s1_ex - s1_ey) / SQRT2
        lo2 = min(s2_sx - s2_sy, s2_ex - s2_ey) / SQRT2
        hi2 = max(s2_sx - s2_sy, s2_ex - s2_ey) / SQRT2
        overlap = min(hi1, hi2) - max(lo1, lo2)
        if overlap < 1e-9:
            return 0.0
        return overlap / max(perp, min_spacing)


@njit(cache=True)
def _point_seg_dist_jit(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-18:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    rx = x1 + t * dx
    ry = y1 + t * dy
    return math.sqrt((px - rx) ** 2 + (py - ry) ** 2)


@njit(cache=True)
def pair_term_jit(wire_width, min_spacing, max_coupling_dist,
                  beta, crossing_penalty, drc_penalty,
                  segs_a, segs_b, obs_a, obs_b):
    """Full pair interaction: crossings + crosstalk + DRC, all in one pass."""
    na = segs_a.shape[0]
    nb_ = segs_b.shape[0]
    noa = obs_a.shape[0]
    nob = obs_b.shape[0]

    xc = 0
    xt = 0.0
    drc = 0

    half_w = wire_width * 0.5

    for i in range(na):
        a_sx = segs_a[i, 0]
        a_sy = segs_a[i, 1]
        a_ex = segs_a[i, 2]
        a_ey = segs_a[i, 3]
        a_ly = segs_a[i, 4]
        a_dir = _get_dir(a_sx, a_sy, a_ex, a_ey)

        for j in range(nb_):
            b_sx = segs_b[j, 0]
            b_sy = segs_b[j, 1]
            b_ex = segs_b[j, 2]
            b_ey = segs_b[j, 3]
            b_ly = segs_b[j, 4]

            if a_ly != b_ly:
                continue

            if _seg_intersect_jit(a_sx, a_sy, a_ex, a_ey,
                                  b_sx, b_sy, b_ex, b_ey):
                xc += 1

            b_dir = _get_dir(b_sx, b_sy, b_ex, b_ey)
            if a_dir == b_dir:
                xt += _coupling_jit(a_sx, a_sy, a_ex, a_ey,
                                    b_sx, b_sy, b_ex, b_ey,
                                    a_dir, min_spacing, max_coupling_dist)

        for j in range(nob):
            if a_ly < obs_b[j, 3] or a_ly > obs_b[j, 4]:
                continue
            dist = _point_seg_dist_jit(obs_b[j, 0], obs_b[j, 1],
                                       a_sx, a_sy, a_ex, a_ey)
            if dist < half_w + obs_b[j, 2] * 0.5 - 1e-9:
                drc += 1

    for i in range(nb_):
        b_sx = segs_b[i, 0]
        b_sy = segs_b[i, 1]
        b_ex = segs_b[i, 2]
        b_ey = segs_b[i, 3]
        b_ly = segs_b[i, 4]
        for j in range(noa):
            if b_ly < obs_a[j, 3] or b_ly > obs_a[j, 4]:
                continue
            dist = _point_seg_dist_jit(obs_a[j, 0], obs_a[j, 1],
                                       b_sx, b_sy, b_ex, b_ey)
            if dist < half_w + obs_a[j, 2] * 0.5 - 1e-9:
                drc += 1

    for i in range(noa):
        for j in range(nob):
            if obs_a[i, 4] < obs_b[j, 3] or obs_b[j, 4] < obs_a[i, 3]:
                continue
            half = (obs_a[i, 2] + obs_b[j, 2]) * 0.5
            if (abs(obs_a[i, 0] - obs_b[j, 0]) < half - 1e-9 and
                    abs(obs_a[i, 1] - obs_b[j, 1]) < half - 1e-9):
                drc += 1

    return crossing_penalty * xc + drc_penalty * drc + beta * xt
