"""Parse benchmark input files (.gp, .iopad, .net, .bump)."""

import os
from typing import Dict, List, Tuple
from .data_types import DesignParams, Pad, Net, Package


def parse_gp(filepath: str) -> DesignParams:
    """Parse the global parameter file."""
    raw: Dict[str, float] = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            key = line.split("]")[0][1:]
            value = line.split("]")[1].strip()
            raw[key] = float(value)

    spacing = raw.get("SPACING", raw.get("WIREWIDTH", 4))
    return DesignParams(
        pkg_width=raw["PACKAGEWIDTH"],
        pkg_height=raw["PACKAGEHEIGHT"],
        num_layers=int(raw["LAYERNUMBER"]),
        wire_width=raw["WIREWIDTH"],
        via_width=raw["VIAWIDTH"],
        pitch=raw.get("PITCH", 30),
        spacing=spacing,
    )


def parse_iopad(filepath: str) -> Dict[str, Dict[str, Pad]]:
    """Parse the IO pad file. Returns {die_name: {pad_name: Pad}}."""
    dies: Dict[str, Dict[str, Pad]] = {}
    current_die = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[DIE]"):
                current_die = line.split("]")[1].strip()
                dies[current_die] = {}
            else:
                parts = line.split()
                if len(parts) >= 5 and current_die is not None:
                    name = parts[0]
                    bbox = (float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))
                    dies[current_die][name] = Pad(name=name, die=current_die, bbox=bbox)
    return dies


def parse_net(filepath: str) -> List[Tuple[str, str, str]]:
    """Parse the net file. Returns [(net_name, pad1_name, pad2_name), ...]."""
    nets = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                nets.append((parts[0], parts[1], parts[2]))
    return nets


def parse_bump(filepath: str) -> List[Tuple[float, float, float, float]]:
    """Parse the bump file. Returns [(x1, y1, x2, y2), ...]."""
    bumps = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                bumps.append((float(parts[0]), float(parts[1]),
                              float(parts[2]), float(parts[3])))
    return bumps


def load_benchmark(benchmark_dir: str) -> Package:
    """Load a complete benchmark from directory."""
    name = os.path.basename(os.path.normpath(benchmark_dir))

    gp_file = os.path.join(benchmark_dir, f"{name}.gp")
    iopad_file = os.path.join(benchmark_dir, f"{name}.iopad")
    net_file = os.path.join(benchmark_dir, f"{name}.net")
    bump_file = os.path.join(benchmark_dir, f"{name}.bump")

    for fpath in [gp_file, iopad_file, net_file, bump_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing file: {fpath}")

    params = parse_gp(gp_file)
    dies = parse_iopad(iopad_file)
    raw_nets = parse_net(net_file)
    bumps = parse_bump(bump_file)

    all_pads: Dict[str, Pad] = {}
    for die_pads in dies.values():
        all_pads.update(die_pads)

    nets: List[Net] = []
    for idx, (net_name, p1_name, p2_name) in enumerate(raw_nets):
        if p1_name not in all_pads or p2_name not in all_pads:
            print(f"Warning: net {net_name} references unknown pad, skipping")
            continue
        pad1 = all_pads[p1_name]
        pad2 = all_pads[p2_name]
        nets.append(Net(idx=idx, name=net_name,
                        pad1_name=p1_name, pad2_name=p2_name,
                        pad1=pad1, pad2=pad2))

    return Package(params=params, dies=dies, nets=nets,
                   bumps=bumps, all_pads=all_pads)
