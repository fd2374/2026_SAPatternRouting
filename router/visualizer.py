"""Visualize routing results with matplotlib, using physical wire/via dimensions."""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from typing import List, Tuple, Optional

from .data_types import (
    RoutingSolution, RoutePattern, Package, Segment, ViaObstacle,
)
from .util import (
    segments_intersect, get_via_obstacles,
    _wire_via_overlap, _via_via_overlap,
)


LAYER_COLORS = [
    "#E63946",  # Layer 0: red
    "#457B9D",  # Layer 1: blue
    "#2A9D8F",  # Layer 2: teal
    "#E9C46A",  # Layer 3: yellow
    "#F4A261",  # Layer 4: orange
    "#264653",  # Layer 5: dark
]

DIE_COLORS = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
              "#A6D854", "#FFD92F"]


# ============================================================
# Violation location finders
# ============================================================

def _seg_intersection_point(s1: Segment, s2: Segment):
    """Compute the intersection point of two segments (assumes they intersect)."""
    ax, ay = s1.start.x, s1.start.y
    bx, by = s1.end.x, s1.end.y
    cx, cy = s2.start.x, s2.start.y
    dx, dy = s2.end.x, s2.end.y

    denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(denom) < 1e-12:
        return ((ax + bx + cx + dx) / 4.0, (ay + by + cy + dy) / 4.0)
    t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom
    return (ax + t * (bx - ax), ay + t * (by - ay))


def _find_crossing_points(routes: List[RoutePattern]):
    """Find all same-layer crossing locations. Returns [(x, y, layer), ...]."""
    pts = []
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            ra, rb = routes[i], routes[j]
            if ra.layers_used.isdisjoint(rb.layers_used):
                continue
            for sa in ra.segments:
                for sb in rb.segments:
                    if segments_intersect(sa, sb):
                        x, y = _seg_intersection_point(sa, sb)
                        pts.append((x, y, sa.layer))
    return pts


def _find_drc_locations(solution: RoutingSolution):
    """
    Find all DRC violation locations.
    Returns two lists:
      wire_via_locs: [(via_x, via_y, layer), ...] for wire-via overlaps
      via_via_locs:  [(mid_x, mid_y, layer), ...] for via-via overlaps
    """
    routes = solution.routes
    nets = solution.package.nets
    params = solution.package.params
    n = len(routes)
    all_obs = [get_via_obstacles(routes[i], nets[i], params) for i in range(n)]

    wv_locs = []
    vv_locs = []

    for i in range(n):
        for j in range(i + 1, n):
            for seg in routes[i].segments:
                for vob in all_obs[j]:
                    if _wire_via_overlap(seg, vob, params.wire_width):
                        shared = {seg.layer} & vob.layers
                        ly = min(shared) if shared else seg.layer
                        wv_locs.append((vob.center.x, vob.center.y, ly))

            for seg in routes[j].segments:
                for vob in all_obs[i]:
                    if _wire_via_overlap(seg, vob, params.wire_width):
                        shared = {seg.layer} & vob.layers
                        ly = min(shared) if shared else seg.layer
                        wv_locs.append((vob.center.x, vob.center.y, ly))

            for va in all_obs[i]:
                for vb in all_obs[j]:
                    if _via_via_overlap(va, vb):
                        shared = va.layers & vb.layers
                        ly = min(shared) if shared else 0
                        mx = (va.center.x + vb.center.x) / 2.0
                        my = (va.center.y + vb.center.y) / 2.0
                        vv_locs.append((mx, my, ly))

    return wv_locs, vv_locs


def _draw_violations(ax, crossing_pts, wv_locs, vv_locs,
                     marker_size, layer_filter=None, zorder=20):
    """
    Draw violation markers on the axes.
      - Crossings: red "X" with a thin circle
      - Wire-via DRC: orange triangle
      - Via-via DRC: magenta square
    marker_size is in data coordinates (half-side of the marker area).
    """
    ms = marker_size

    for x, y, ly in crossing_pts:
        if layer_filter is not None and ly != layer_filter:
            continue
        ax.plot([x - ms, x + ms], [y - ms, y + ms],
                color="#FF0000", linewidth=0.6, zorder=zorder)
        ax.plot([x - ms, x + ms], [y + ms, y - ms],
                color="#FF0000", linewidth=0.6, zorder=zorder)
        circ = patches.Circle((x, y), ms * 1.3, linewidth=0.4,
                               edgecolor="#FF0000", facecolor="none",
                               zorder=zorder)
        ax.add_patch(circ)

    for x, y, ly in wv_locs:
        if layer_filter is not None and ly != layer_filter:
            continue
        tri = patches.RegularPolygon(
            (x, y), numVertices=3, radius=ms * 1.4,
            orientation=0, linewidth=0.4,
            edgecolor="#FF8C00", facecolor="#FF8C00",
            alpha=0.7, zorder=zorder,
        )
        ax.add_patch(tri)

    for x, y, ly in vv_locs:
        if layer_filter is not None and ly != layer_filter:
            continue
        rect = patches.Rectangle(
            (x - ms * 0.8, y - ms * 0.8), ms * 1.6, ms * 1.6,
            linewidth=0.4, edgecolor="#CC00CC", facecolor="#CC00CC",
            alpha=0.6, zorder=zorder,
        )
        ax.add_patch(rect)


# ============================================================
# Drawing helpers
# ============================================================

def _seg_to_rect(sx, sy, ex, ey, half_w):
    """Convert a segment centerline to a 4-corner rectangle polygon."""
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return None
    nx = -dy / length * half_w
    ny = dx / length * half_w
    return [
        (sx + nx, sy + ny),
        (ex + nx, ey + ny),
        (ex - nx, ey - ny),
        (sx - nx, sy - ny),
    ]


def _draw_wires(ax, routes, layer, color, params, alpha=0.7, zorder=5):
    """Draw wire segments as rectangles with physical WIREWIDTH in data coords."""
    half_w = params.wire_width / 2.0
    polys = []
    for route in routes:
        for seg in route.segments:
            if seg.layer != layer:
                continue
            rect = _seg_to_rect(seg.start.x, seg.start.y,
                                seg.end.x, seg.end.y, half_w)
            if rect:
                polys.append(rect)
    if not polys:
        return 0
    pc = PolyCollection(polys, facecolors=color, edgecolors="none",
                        alpha=alpha, zorder=zorder)
    ax.add_collection(pc)
    return len(polys)


def _draw_vias(ax, routes, params, layer_filter=None, zorder=6):
    """Draw route vias as VIAWIDTH squares in data coordinates."""
    vw = params.via_width
    half = vw / 2.0
    for route in routes:
        for via in route.vias:
            lo = min(via.from_layer, via.to_layer)
            hi = max(via.from_layer, via.to_layer)
            if layer_filter is not None and not (lo <= layer_filter <= hi):
                continue

            c_from = LAYER_COLORS[via.from_layer % len(LAYER_COLORS)]
            c_to = LAYER_COLORS[via.to_layer % len(LAYER_COLORS)]

            rect = patches.Rectangle(
                (via.pos.x - half, via.pos.y - half), vw, vw,
                linewidth=0.6, edgecolor="black", facecolor="white",
                alpha=0.9, zorder=zorder,
            )
            ax.add_patch(rect)

            ax.plot([via.pos.x - half * 0.5, via.pos.x + half * 0.5],
                    [via.pos.y, via.pos.y],
                    color=c_from, linewidth=0.5, zorder=zorder + 1)
            ax.plot([via.pos.x, via.pos.x],
                    [via.pos.y - half * 0.5, via.pos.y + half * 0.5],
                    color=c_to, linewidth=0.5, zorder=zorder + 1)


def _draw_pad_vias(ax, solution, params, layer_filter=None, zorder=5):
    """
    Draw pad-via stacks at each net endpoint.
    Drawn as VIAWIDTH squares with a diamond marker.
    """
    vw = params.via_width
    half = vw / 2.0
    routes = solution.routes
    nets = solution.package.nets

    for net_idx, route in enumerate(routes):
        if not route.segments:
            continue
        net = nets[net_idx]

        for pt, conn_layer in [(net.pt1, route.segments[0].layer),
                               (net.pt2, route.segments[-1].layer)]:
            if layer_filter is not None and layer_filter > conn_layer:
                continue

            c = LAYER_COLORS[conn_layer % len(LAYER_COLORS)]

            rect = patches.Rectangle(
                (pt.x - half, pt.y - half), vw, vw,
                linewidth=0.6, edgecolor=c, facecolor=c,
                alpha=0.5, zorder=zorder,
            )
            ax.add_patch(rect)

            d = half * 0.4
            diamond = patches.Polygon(
                [(pt.x, pt.y + d), (pt.x + d, pt.y),
                 (pt.x, pt.y - d), (pt.x - d, pt.y)],
                closed=True, facecolor="white", edgecolor=c,
                linewidth=0.4, alpha=0.9, zorder=zorder + 1,
            )
            ax.add_patch(diamond)


# ============================================================
# Plot functions
# ============================================================

def save_blocked_grid(blocked, params, grid_step, nx, ny, out_dir):
    """Save per-layer blocked grid images."""
    os.makedirs(out_dir, exist_ok=True)
    nl = blocked.shape[0]
    for layer in range(nl):
        fig, ax = plt.subplots(1, 1,
            figsize=(10, 10 * params.pkg_height / params.pkg_width))
        ax.imshow(blocked[layer], origin='lower', cmap='Greys',
                  extent=[0, (nx - 1) * grid_step, 0, (ny - 1) * grid_step],
                  aspect='equal', interpolation='nearest')
        ax.set_title(f"Blocked Grid — Layer {layer}  (step={grid_step})",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        p = os.path.join(out_dir, f"blocked_grid_L{layer}.png")
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [A*] Saved {p}")


def visualize_solution(
    solution: RoutingSolution,
    output_dir: str,
    show: bool = True,
):
    """Generate all visualization plots for a routing solution."""
    os.makedirs(output_dir, exist_ok=True)

    crossing_pts = _find_crossing_points(solution.routes)
    wv_locs, vv_locs = _find_drc_locations(solution)

    _plot_overview(solution, output_dir, show, crossing_pts, wv_locs, vv_locs)
    _plot_per_layer(solution, output_dir, show, crossing_pts, wv_locs, vv_locs)
    _plot_crosstalk_heatmap(solution, output_dir, show)


def _plot_overview(solution, output_dir, show,
                   crossing_pts, wv_locs, vv_locs):
    """Overview plot showing all routes with violation markers."""
    pkg = solution.package
    params = pkg.params
    routes = solution.routes

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.set_title(
        f"Routing Overview ({len(pkg.nets)} nets, {params.num_layers} layers)\n"
        f"WIREWIDTH={params.wire_width}  VIAWIDTH={params.via_width}  "
        f"Xings={solution.num_crossings}  DRC={solution.num_drc_violations}",
        fontsize=13, fontweight="bold",
    )
    ax.set_aspect("equal")

    ax.add_patch(patches.Rectangle(
        (0, 0), params.pkg_width, params.pkg_height,
        linewidth=2, edgecolor="black", facecolor="#f8f8f8", zorder=1))

    if pkg.bumps:
        bx = [(b[0] + b[2]) / 2 for b in pkg.bumps]
        by = [(b[1] + b[3]) / 2 for b in pkg.bumps]
        ms = max(0.3, min(2, 4000 / len(pkg.bumps)))
        ax.scatter(bx, by, s=ms, c="#dddddd", marker="s", zorder=2)

    for i, (die_name, pads) in enumerate(sorted(pkg.dies.items())):
        color = DIE_COLORS[i % len(DIE_COLORS)]
        _draw_die_bbox(ax, die_name, pads, color, zorder=3)

    for die_name, pads in pkg.dies.items():
        for pad in pads.values():
            x1, y1, x2, y2 = pad.bbox
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=0.3, edgecolor="gray", facecolor="#aaaaaa",
                alpha=0.5, zorder=4))

    ax.set_xlim(-params.pkg_width * 0.03, params.pkg_width * 1.03)
    ax.set_ylim(-params.pkg_height * 0.03, params.pkg_height * 1.03)

    legend_handles = []
    for layer in range(params.num_layers):
        color = LAYER_COLORS[layer % len(LAYER_COLORS)]
        n_segs = _draw_wires(ax, routes, layer, color, params, zorder=5 + layer)
        if n_segs > 0:
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=3, label=f"Layer {layer}"))

    _draw_pad_vias(ax, solution, params, zorder=5 + params.num_layers)
    _draw_vias(ax, routes, params, zorder=5 + params.num_layers + 1)

    # Violation markers — size proportional to package so they're visible
    vm_size = max(params.pkg_width, params.pkg_height) * 0.006
    _draw_violations(ax, crossing_pts, wv_locs, vv_locs,
                     marker_size=vm_size, zorder=15)

    legend_handles.append(
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Route via"))
    legend_handles.append(
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markeredgecolor="gray", markersize=6, label="Pad via"))
    if crossing_pts:
        legend_handles.append(
            Line2D([0], [0], marker="x", color="#FF0000", markersize=7,
                   linestyle="none", label=f"Crossing ({len(crossing_pts)})"))
    if wv_locs:
        legend_handles.append(
            Line2D([0], [0], marker="^", color="#FF8C00", markersize=7,
                   linestyle="none", label=f"Wire-via DRC ({len(wv_locs)})"))
    if vv_locs:
        legend_handles.append(
            Line2D([0], [0], marker="s", color="#CC00CC", markersize=6,
                   linestyle="none", label=f"Via-via DRC ({len(vv_locs)})"))

    ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
              framealpha=0.9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    path = os.path.join(output_dir, "routing_overview.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[Viz] Saved {path}")
    if show:
        plt.show()
    plt.close()


def _plot_per_layer(solution, output_dir, show,
                    crossing_pts, wv_locs, vv_locs):
    """One subplot per layer with violation markers."""
    pkg = solution.package
    params = pkg.params
    routes = solution.routes
    n_layers = params.num_layers

    cols = min(n_layers, 3)
    rows = math.ceil(n_layers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 9 * rows),
                             squeeze=False)
    fig.suptitle("Per-Layer Routing View (physical dimensions)", fontsize=16,
                 fontweight="bold")

    vm_size = max(params.pkg_width, params.pkg_height) * 0.006

    for layer in range(n_layers):
        r, c = divmod(layer, cols)
        ax = axes[r][c]
        color = LAYER_COLORS[layer % len(LAYER_COLORS)]

        ax.set_title(f"Layer {layer}", fontsize=13, fontweight="bold",
                     color=color)
        ax.set_aspect("equal")

        ax.add_patch(patches.Rectangle(
            (0, 0), params.pkg_width, params.pkg_height,
            linewidth=1.5, edgecolor="black", facecolor="#fafafa", zorder=1))

        for i, (die_name, pads) in enumerate(sorted(pkg.dies.items())):
            dc = DIE_COLORS[i % len(DIE_COLORS)]
            _draw_die_bbox(ax, die_name, pads, dc, zorder=2)

        ax.set_xlim(-params.pkg_width * 0.03, params.pkg_width * 1.03)
        ax.set_ylim(-params.pkg_height * 0.03, params.pkg_height * 1.03)

        n_segs = _draw_wires(ax, routes, layer, color, params,
                             alpha=0.75, zorder=5)

        _draw_pad_vias(ax, solution, params, layer_filter=layer, zorder=6)
        _draw_vias(ax, routes, params, layer_filter=layer, zorder=7)

        _draw_violations(ax, crossing_pts, wv_locs, vv_locs,
                         marker_size=vm_size, layer_filter=layer, zorder=15)

        n_xings = sum(1 for _, _, ly in crossing_pts if ly == layer)
        n_wv = sum(1 for _, _, ly in wv_locs if ly == layer)
        n_vv = sum(1 for _, _, ly in vv_locs if ly == layer)

        n_pvias = 0
        nets = solution.package.nets
        for net_idx, route in enumerate(routes):
            if not route.segments:
                continue
            if route.segments[0].layer >= layer:
                n_pvias += 1
            if route.segments[-1].layer >= layer:
                n_pvias += 1

        n_rvias = 0
        for route in routes:
            for via in route.vias:
                lo = min(via.from_layer, via.to_layer)
                hi = max(via.from_layer, via.to_layer)
                if lo <= layer <= hi:
                    n_rvias += 1

        info = f"{n_segs} segs, {n_rvias} vias, {n_pvias} pad-vias"
        if n_xings or n_wv or n_vv:
            info += f"\nXings={n_xings}  WV-DRC={n_wv}  VV-DRC={n_vv}"
        ax.text(0.02, 0.98, info,
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        ax.grid(True, alpha=0.15)

    for idx in range(n_layers, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "routing_per_layer.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[Viz] Saved {path}")
    if show:
        plt.show()
    plt.close()


def _plot_crosstalk_heatmap(solution, output_dir, show):
    """Heatmap of pairwise crosstalk between nets."""
    from .util import route_pair_crosstalk

    params = solution.package.params
    nets = solution.package.nets
    routes = solution.routes
    n = len(nets)

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            xt = route_pair_crosstalk(routes[i], routes[j], params)
            matrix[i][j] = xt
            matrix[j][i] = xt

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_title("Pairwise Crosstalk Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Net index")
    ax.set_ylabel("Net index")

    if n <= 30:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([net.name for net in nets], rotation=90, fontsize=7)
        ax.set_yticklabels([net.name for net in nets], fontsize=7)

    fig.colorbar(im, ax=ax, label="Crosstalk")
    plt.tight_layout()

    path = os.path.join(output_dir, "crosstalk_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[Viz] Saved {path}")
    if show:
        plt.show()
    plt.close()


class SALivePlot:
    """Live-updating SA convergence plot (Energy, Temperature, Acceptance Rate)."""

    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(
            3, 1, figsize=(9, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1]})

        ax_e, ax_t, ax_a = self.axes

        ax_e.set_title("SA Convergence", fontsize=13, fontweight="bold")
        ax_e.set_ylabel("Energy")
        ax_e.grid(True, alpha=0.2)
        self.line_cur, = ax_e.plot([], [], lw=0.8, alpha=0.6,
                                   color="#457B9D", label="Current")
        self.line_best, = ax_e.plot([], [], lw=1.4,
                                    color="#E63946", label="Best")
        ax_e.legend(loc="upper right", fontsize=9)

        ax_t.set_ylabel("Temperature")
        ax_t.grid(True, alpha=0.2)
        self.line_temp, = ax_t.plot([], [], lw=1.0, color="#E9C46A")

        ax_a.set_ylabel("Accept Rate")
        ax_a.set_xlabel("Iteration")
        ax_a.set_ylim(-0.05, 1.05)
        ax_a.grid(True, alpha=0.2)
        self.line_acc, = ax_a.plot([], [], lw=0.8, color="#2A9D8F")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self._iters = []
        self._energy = []
        self._best = []
        self._temp = []
        self._acc = []

    def update(self, iteration, energy, best_energy, temperature, accept_rate):
        self._iters.append(iteration)
        self._energy.append(energy)
        self._best.append(best_energy)
        self._temp.append(temperature)
        self._acc.append(accept_rate)

        self.line_cur.set_data(self._iters, self._energy)
        self.line_best.set_data(self._iters, self._best)
        self.line_temp.set_data(self._iters, self._temp)
        self.line_acc.set_data(self._iters, self._acc)

        xmax = max(self._iters[-1], 2)
        n = len(self._energy)

        ax_e = self.axes[0]
        skip = max(1, n * 3 // 4)
        tail_e = self._energy[skip:]
        tail_b = self._best[skip:]
        if tail_e:
            margin = max((max(tail_e) - min(tail_b)) * 0.1, 1.0)
            ax_e.set_ylim(min(tail_b) - margin, max(tail_e) + margin)
        ax_e.set_xlim(1, xmax)

        for ax in self.axes[1:]:
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(1, xmax)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def mark_reheat(self, iteration):
        """Draw a vertical dashed line at a reheat event."""
        for ax in self.axes:
            ax.axvline(x=iteration, color="#E63946", ls="--",
                       lw=0.6, alpha=0.5)

    def finish(self):
        plt.ioff()


def _draw_die_bbox(ax, die_name, pads, color, zorder=3):
    """Draw a die bounding box around its IO pads."""
    if not pads:
        return
    xs, ys = [], []
    for pad in pads.values():
        xs.extend([pad.bbox[0], pad.bbox[2]])
        ys.extend([pad.bbox[1], pad.bbox[3]])

    margin_x = (max(xs) - min(xs)) * 0.05 + 30
    margin_y = (max(ys) - min(ys)) * 0.05 + 30
    x0 = min(xs) - margin_x
    y0 = min(ys) - margin_y
    w = max(xs) - min(xs) + 2 * margin_x
    h = max(ys) - min(ys) + 2 * margin_y

    ax.add_patch(patches.FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0",
        linewidth=1.5, edgecolor=color,
        facecolor=(*plt.cm.colors.to_rgba(color)[:3], 0.08),
        zorder=zorder))

    cx, cy = x0 + w / 2, y0 + h / 2
    ax.text(cx, cy, die_name, fontsize=10, fontweight="bold",
            ha="center", va="center", color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.8),
            zorder=zorder + 7)
