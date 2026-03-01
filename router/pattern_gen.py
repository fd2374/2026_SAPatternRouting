"""
135-degree (obtuse-angle only) pattern generation for 2-pin nets.

Allowed directions: Horizontal, Vertical, 45-degree diagonals.
All turns must be 135 degrees (obtuse), meaning:
  - H <-> Diagonal is allowed (135 deg)
  - V <-> Diagonal is allowed (135 deg)
  - H <-> V is FORBIDDEN (90 deg)
  - Diagonal <-> orthogonal Diagonal is FORBIDDEN (90 deg)

Pattern types for a 2-pin net from A to B:
  Type 1 "Homo":  Rect-Diag-Rect with same rectilinear direction
    - H-D-H (horizontal bookends) for |dx| >= |dy|
    - V-D-V (vertical bookends)   for |dy| >= |dx|
  Type 2 "Mixed": Rect-Diag-Rect with different rectilinear directions
    - V-D-H: vertical start, diagonal, horizontal end
    - H-D-V: horizontal start, diagonal, vertical end
    These produce fundamentally different geometric shapes, crucial for
    avoiding same-layer crossings between nets.
"""

import math
from typing import List
from .data_types import Point, Segment, Via, RoutePattern


SQRT2 = math.sqrt(2.0)

HOMO_SPLIT_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
MIXED_DIAG_FRACTIONS = [0.2, 0.5, 0.8, 1.0]
VIA_SPLIT_RATIOS = [0.25, 0.5, 0.75]


def _sign(x: float) -> float:
    if x > 1e-9:
        return 1.0
    elif x < -1e-9:
        return -1.0
    return 0.0


def _build_segments(points: List[Point], layer: int) -> List[Segment]:
    """Build segments from a list of points, skipping zero-length ones."""
    segments = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        if p1.distance_to(p2) > 1e-9:
            segments.append(Segment(start=p1, end=p2, layer=layer))
    return segments


def _compute_wirelength(points: List[Point]) -> float:
    wl = 0.0
    for i in range(len(points) - 1):
        wl += points[i].distance_to(points[i + 1])
    return wl


# ============================================================
# Type 1: Homogeneous patterns (H-D-H or V-D-V)
# ============================================================

def _gen_homo_pattern(
    pt1: Point, pt2: Point, layer: int, split_ratio: float,
    net_idx: int, pattern_idx: int,
) -> RoutePattern:
    """
    Monotone shortest-path pattern: Rect(r*R) -> Diag(D) -> Rect((1-r)*R)
    where Rect is the dominant rectilinear direction.
    """
    dx = pt2.x - pt1.x
    dy = pt2.y - pt1.y
    adx, ady = abs(dx), abs(dy)
    sx, sy = _sign(dx), _sign(dy)

    if adx < 1e-9 and ady < 1e-9:
        return RoutePattern(net_idx=net_idx, pattern_idx=pattern_idx,
                            segments=[], vias=[])
    if ady < 1e-9:
        return RoutePattern(net_idx=net_idx, pattern_idx=pattern_idx,
                            segments=[Segment(pt1, pt2, layer)], vias=[])
    if adx < 1e-9:
        return RoutePattern(net_idx=net_idx, pattern_idx=pattern_idx,
                            segments=[Segment(pt1, pt2, layer)], vias=[])
    if abs(adx - ady) < 1e-9:
        return RoutePattern(net_idx=net_idx, pattern_idx=pattern_idx,
                            segments=[Segment(pt1, pt2, layer)], vias=[])

    diag_dist = min(adx, ady)
    rect_dist = max(adx, ady) - diag_dist
    is_h_dominant = adx >= ady

    r1 = split_ratio * rect_dist
    r2 = (1.0 - split_ratio) * rect_dist

    if is_h_dominant:
        p0 = pt1
        p1 = Point(pt1.x + sx * r1, pt1.y)
        p2 = Point(p1.x + sx * diag_dist, p1.y + sy * diag_dist)
        p3 = pt2
    else:
        p0 = pt1
        p1 = Point(pt1.x, pt1.y + sy * r1)
        p2 = Point(p1.x + sx * diag_dist, p1.y + sy * diag_dist)
        p3 = pt2

    points = [p0, p1, p2, p3]
    segments = _build_segments(points, layer)
    wirelength = _compute_wirelength(points)
    return RoutePattern(net_idx=net_idx, pattern_idx=pattern_idx,
                        segments=segments, vias=[], wirelength=wirelength)


# ============================================================
# Type 1b: Inverted patterns (D-H-D or D-V-D)
# ============================================================

def _gen_drd_pattern(
    pt1: Point, pt2: Point, layer: int, split_ratio: float,
    net_idx: int, pattern_idx: int,
) -> RoutePattern:
    """
    Inverted shortest-path pattern: Diag(r*D) -> Rect(R) -> Diag((1-r)*D)
    where Rect is the dominant rectilinear direction.

    Same wirelength as the homo pattern but geometrically different:
    the diagonal segments are on the outside instead of the inside.
    At split_ratio 0.0 or 1.0 this produces 2-segment L-shaped routes.
    """
    dx = pt2.x - pt1.x
    dy = pt2.y - pt1.y
    adx, ady = abs(dx), abs(dy)
    sx, sy = _sign(dx), _sign(dy)

    if adx < 1e-9 or ady < 1e-9:
        return None
    if abs(adx - ady) < 1e-9:
        return None

    diag_total = min(adx, ady)
    rect_dist = max(adx, ady) - diag_total
    is_h_dominant = adx >= ady

    d1 = split_ratio * diag_total

    if is_h_dominant:
        p0 = pt1
        p1 = Point(pt1.x + sx * d1, pt1.y + sy * d1)
        p2 = Point(p1.x + sx * rect_dist, p1.y)
        p3 = pt2
    else:
        p0 = pt1
        p1 = Point(pt1.x + sx * d1, pt1.y + sy * d1)
        p2 = Point(p1.x, p1.y + sy * rect_dist)
        p3 = pt2

    points = [p0, p1, p2, p3]
    segments = _build_segments(points, layer)
    wirelength = _compute_wirelength(points)
    return RoutePattern(net_idx=net_idx, pattern_idx=pattern_idx,
                        segments=segments, vias=[], wirelength=wirelength)


# ============================================================
# Type 2: Mixed patterns (V-D-H and H-D-V)
# ============================================================

def _gen_vdh_pattern(
    pt1: Point, pt2: Point, layer: int, diag_frac: float,
    net_idx: int, pattern_idx: int,
) -> RoutePattern:
    """
    V-D-H pattern: Vertical -> Diagonal -> Horizontal.
    diag_frac in (0, 1]: fraction of min(|dx|,|dy|) used as diagonal.
      At diag_frac=1.0 the diagonal covers all of min(|dx|,|dy|),
      giving the minimum wirelength (same as homo pattern).
      Smaller diag_frac = more wirelength but very different geometry.

    V(v) -> D(d) -> H(h)
    where: d = diag_frac * min(|dx|, |dy|)
           v = |dy| - d     (remaining vertical)
           h = |dx| - d     (remaining horizontal)
    """
    dx = pt2.x - pt1.x
    dy = pt2.y - pt1.y
    adx, ady = abs(dx), abs(dy)
    sx, sy = _sign(dx), _sign(dy)

    if adx < 1e-9 or ady < 1e-9:
        return None  # degenerate, skip

    d = diag_frac * min(adx, ady)
    v = ady - d
    h = adx - d

    if v < -1e-9 or h < -1e-9:
        return None

    p0 = pt1
    p1 = Point(pt1.x, pt1.y + sy * v)            # after vertical
    p2 = Point(p1.x + sx * d, p1.y + sy * d)     # after diagonal
    p3 = pt2                                       # should match p2 + horizontal

    points = [p0, p1, p2, p3]
    segments = _build_segments(points, layer)
    wirelength = _compute_wirelength(points)
    return RoutePattern(net_idx=net_idx, pattern_idx=pattern_idx,
                        segments=segments, vias=[], wirelength=wirelength)


def _gen_hdv_pattern(
    pt1: Point, pt2: Point, layer: int, diag_frac: float,
    net_idx: int, pattern_idx: int,
) -> RoutePattern:
    """
    H-D-V pattern: Horizontal -> Diagonal -> Vertical.
    Mirror of V-D-H.
    """
    dx = pt2.x - pt1.x
    dy = pt2.y - pt1.y
    adx, ady = abs(dx), abs(dy)
    sx, sy = _sign(dx), _sign(dy)

    if adx < 1e-9 or ady < 1e-9:
        return None

    d = diag_frac * min(adx, ady)
    h = adx - d
    v = ady - d

    if v < -1e-9 or h < -1e-9:
        return None

    p0 = pt1
    p1 = Point(pt1.x + sx * h, pt1.y)            # after horizontal
    p2 = Point(p1.x + sx * d, p1.y + sy * d)     # after diagonal
    p3 = pt2

    points = [p0, p1, p2, p3]
    segments = _build_segments(points, layer)
    wirelength = _compute_wirelength(points)
    return RoutePattern(net_idx=net_idx, pattern_idx=pattern_idx,
                        segments=segments, vias=[], wirelength=wirelength)


# ============================================================
# Multi-layer: split any single-layer pattern across two layers
# ============================================================

def _split_to_multilayer(
    base_pattern: RoutePattern,
    layer1: int, layer2: int,
    split_after_seg: int,
    net_idx: int, pattern_idx: int,
) -> RoutePattern:
    """
    Take a single-layer pattern and split it across two layers.
    Segments [0..split_after_seg] on layer1, rest on layer2, via at junction.
    """
    segs = base_pattern.segments
    if not segs or split_after_seg < 0 or split_after_seg >= len(segs):
        return None

    new_segs = []
    for i, s in enumerate(segs):
        lyr = layer1 if i <= split_after_seg else layer2
        new_segs.append(Segment(s.start, s.end, lyr))

    via_pos = segs[split_after_seg].end
    vias = [Via(pos=via_pos, from_layer=layer1, to_layer=layer2)]

    return RoutePattern(net_idx=net_idx, pattern_idx=pattern_idx,
                        segments=new_segs, vias=vias,
                        wirelength=base_pattern.wirelength)


def _split_segment_multilayer(
    base_pattern: RoutePattern,
    seg_idx: int,
    via_ratio: float,
    layer1: int, layer2: int,
    net_idx: int, pattern_idx: int,
) -> RoutePattern:
    """
    Split a single-layer pattern by placing a via at an arbitrary point
    along segment seg_idx. The via position is interpolated at via_ratio
    between the segment's start and end.

    Segments [0..seg_idx) and the first part of seg_idx stay on layer1;
    the second part of seg_idx and [seg_idx+1..] go to layer2.
    """
    segs = base_pattern.segments
    if not segs or seg_idx < 0 or seg_idx >= len(segs):
        return None

    seg = segs[seg_idx]
    vx = seg.start.x + via_ratio * (seg.end.x - seg.start.x)
    vy = seg.start.y + via_ratio * (seg.end.y - seg.start.y)
    via_pt = Point(vx, vy)

    if seg.start.distance_to(via_pt) < 1e-9 or via_pt.distance_to(seg.end) < 1e-9:
        return None

    new_segs = []
    for i in range(seg_idx):
        new_segs.append(Segment(segs[i].start, segs[i].end, layer1))
    new_segs.append(Segment(seg.start, via_pt, layer1))
    new_segs.append(Segment(via_pt, seg.end, layer2))
    for i in range(seg_idx + 1, len(segs)):
        new_segs.append(Segment(segs[i].start, segs[i].end, layer2))

    vias = [Via(pos=via_pt, from_layer=layer1, to_layer=layer2)]
    return RoutePattern(net_idx=net_idx, pattern_idx=pattern_idx,
                        segments=new_segs, vias=vias,
                        wirelength=base_pattern.wirelength)


# ============================================================
# Main candidate generator
# ============================================================

def generate_candidates(
    net_idx: int,
    pt1: Point, pt2: Point,
    num_layers: int,
    split_ratios: List[float] = None,
    diag_fractions: List[float] = None,
    via_split_ratios: List[float] = None,
    enable_multilayer: bool = True,
) -> List[RoutePattern]:
    """
    Generate all candidate routing patterns for one 2-pin net.

    Includes:
      - Homo patterns (H-D-H / V-D-V) with various split ratios
      - Mixed patterns (V-D-H / H-D-V) with various diagonal fractions
      - Multi-layer variants: via at bend points AND mid-segment positions
    """
    if split_ratios is None:
        split_ratios = HOMO_SPLIT_RATIOS
    if diag_fractions is None:
        diag_fractions = MIXED_DIAG_FRACTIONS
    if via_split_ratios is None:
        via_split_ratios = VIA_SPLIT_RATIOS

    candidates: List[RoutePattern] = []
    pidx = 0

    # --- Single-layer patterns ---
    single_layer_bases: List[RoutePattern] = []

    for layer in range(num_layers):
        # Type 1: Homo (H-D-H or V-D-V)
        for r in split_ratios:
            pat = _gen_homo_pattern(pt1, pt2, layer, r, net_idx, pidx)
            candidates.append(pat)
            single_layer_bases.append(pat)
            pidx += 1

        # Type 1b: Inverted (D-H-D or D-V-D)
        for r in split_ratios:
            pat = _gen_drd_pattern(pt1, pt2, layer, r, net_idx, pidx)
            if pat is not None:
                candidates.append(pat)
                single_layer_bases.append(pat)
                pidx += 1

        # Type 2: Mixed (V-D-H and H-D-V)
        for df in diag_fractions:
            pat = _gen_vdh_pattern(pt1, pt2, layer, df, net_idx, pidx)
            if pat is not None:
                candidates.append(pat)
                single_layer_bases.append(pat)
                pidx += 1

            pat = _gen_hdv_pattern(pt1, pt2, layer, df, net_idx, pidx)
            if pat is not None:
                candidates.append(pat)
                single_layer_bases.append(pat)
                pidx += 1

    # --- Multi-layer patterns ---
    if enable_multilayer and num_layers > 1:
        for base in single_layer_bases:
            if not base.segments:
                continue
            base_layer = base.segments[0].layer
            for other_layer in range(num_layers):
                if other_layer == base_layer:
                    continue

                # Via at bend points (segment boundaries)
                if len(base.segments) >= 2:
                    for split_at in range(len(base.segments) - 1):
                        ml = _split_to_multilayer(
                            base, base_layer, other_layer, split_at,
                            net_idx, pidx)
                        if ml is not None:
                            candidates.append(ml)
                            pidx += 1

                # Via at arbitrary positions along each segment
                for seg_idx in range(len(base.segments)):
                    for vr in via_split_ratios:
                        ml = _split_segment_multilayer(
                            base, seg_idx, vr,
                            base_layer, other_layer,
                            net_idx, pidx)
                        if ml is not None:
                            candidates.append(ml)
                            pidx += 1

    # Deduplicate
    candidates = _deduplicate(candidates)
    for i, c in enumerate(candidates):
        c.pattern_idx = i

    return candidates


def _deduplicate(patterns: List[RoutePattern]) -> List[RoutePattern]:
    """Remove duplicate patterns based on segment geometry."""
    seen = set()
    unique = []
    for p in patterns:
        key = _pattern_key(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _pattern_key(p: RoutePattern) -> tuple:
    """Create a hashable key from pattern geometry."""
    seg_keys = []
    for s in p.segments:
        seg_keys.append((
            round(s.start.x, 1), round(s.start.y, 1),
            round(s.end.x, 1), round(s.end.y, 1),
            s.layer,
        ))
    via_keys = []
    for v in p.vias:
        via_keys.append((
            round(v.pos.x, 1), round(v.pos.y, 1),
            v.from_layer, v.to_layer,
        ))
    return (tuple(seg_keys), tuple(via_keys))
