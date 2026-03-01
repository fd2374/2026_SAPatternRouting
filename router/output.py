"""Write routing results to output files."""

import os
from typing import TextIO
from .data_types import RoutingSolution, RoutePattern
from .util import route_pair_crosstalk


def write_solution(solution: RoutingSolution, output_dir: str):
    """Write routing solution to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    routes_file = os.path.join(output_dir, "routes.txt")
    summary_file = os.path.join(output_dir, "summary.txt")
    xt_file = os.path.join(output_dir, "crosstalk.txt")

    _write_routes(solution, routes_file)
    _write_summary(solution, summary_file)
    _write_crosstalk_detail(solution, xt_file)

    print(f"[Output] Results written to {output_dir}/")


def _write_routes(solution: RoutingSolution, filepath: str):
    """Write per-net route details."""
    with open(filepath, "w") as f:
        f.write(f"# Routing solution: {len(solution.package.nets)} nets\n")
        f.write(f"# Format: [NET] name pad1 pad2\n")
        f.write(f"#         [SEG] x1 y1 x2 y2 layer\n")
        f.write(f"#         [VIA] x y from_layer to_layer\n\n")

        for i, net in enumerate(solution.package.nets):
            route = solution.get_route(i)
            f.write(f"[NET] {net.name} {net.pad1_name} {net.pad2_name}\n")

            for seg in route.segments:
                f.write(f"[SEG] {seg.start.x:.4f} {seg.start.y:.4f} "
                        f"{seg.end.x:.4f} {seg.end.y:.4f} {seg.layer}\n")

            for via in route.vias:
                f.write(f"[VIA] {via.pos.x:.4f} {via.pos.y:.4f} "
                        f"{via.from_layer} {via.to_layer}\n")

            f.write(f"# wirelength={route.wirelength:.2f} "
                    f"vias={route.num_vias} "
                    f"layers={sorted(route.layers_used)}\n\n")


def _write_summary(solution: RoutingSolution, filepath: str):
    """Write solution summary."""
    params = solution.package.params
    with open(filepath, "w") as f:
        f.write("# Routing Solution Summary\n\n")
        f.write(f"PACKAGE_WIDTH      {params.pkg_width}\n")
        f.write(f"PACKAGE_HEIGHT     {params.pkg_height}\n")
        f.write(f"NUM_LAYERS         {params.num_layers}\n")
        f.write(f"WIRE_WIDTH         {params.wire_width}\n")
        f.write(f"SPACING            {params.spacing}\n")
        f.write(f"VIA_WIDTH          {params.via_width}\n\n")
        f.write(f"NUM_NETS           {len(solution.package.nets)}\n")
        f.write(f"TOTAL_WIRELENGTH   {solution.total_wirelength:.2f}\n")
        f.write(f"TOTAL_CROSSTALK    {solution.total_crosstalk:.6f}\n")
        f.write(f"MAX_CROSSTALK      {solution.max_crosstalk:.6f}\n")
        f.write(f"TOTAL_VIAS         {solution.num_vias}\n")
        f.write(f"SAME_LAYER_XINGS   {solution.num_crossings}\n")
        f.write(f"DRC_VIOLATIONS     {solution.num_drc_violations}\n\n")

        # Per-layer statistics
        layer_wl = {}
        layer_segs = {}
        for route in solution.all_routes:
            for seg in route.segments:
                layer_wl[seg.layer] = layer_wl.get(seg.layer, 0) + seg.length
                layer_segs[seg.layer] = layer_segs.get(seg.layer, 0) + 1

        f.write("# Per-layer statistics\n")
        for layer in sorted(layer_wl.keys()):
            f.write(f"LAYER {layer}  wirelength={layer_wl[layer]:.2f}  "
                    f"segments={layer_segs[layer]}\n")


def _write_crosstalk_detail(solution: RoutingSolution, filepath: str):
    """Write detailed pairwise crosstalk information."""
    params = solution.package.params
    nets = solution.package.nets
    routes = solution.all_routes

    with open(filepath, "w") as f:
        f.write("# Pairwise crosstalk between nets\n")
        f.write("# Format: net_i net_j crosstalk_value\n\n")

        pairs = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                xt = route_pair_crosstalk(routes[i], routes[j], params)
                if xt > 1e-12:
                    pairs.append((nets[i].name, nets[j].name, xt))

        pairs.sort(key=lambda t: -t[2])
        for n1, n2, xt in pairs:
            f.write(f"{n1} {n2} {xt:.6f}\n")

        f.write(f"\n# Total non-zero pairs: {len(pairs)}\n")
