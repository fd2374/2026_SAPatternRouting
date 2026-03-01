"""
Power Plane Benchmark Visualizer
可视化封装级多芯片电源平面布线 benchmark

用法:
    python visualize.py <benchmark_dir>
    例如: python visualize.py dense2
          python visualize.py pkg1

功能:
    1. 显示封装边界
    2. 显示 bump pad 阵列
    3. 按 die 分颜色显示 IO pad
    4. 显示 die 的边界框
    5. 显示 chip-to-chip net 连接线
    6. 区分 chip-to-chip 和 chip-to-board IO pad
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import numpy as np


# ============== 解析函数 ==============

def parse_gp(filepath):
    """解析 .gp 文件，返回全局参数字典"""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: [KEY] value
            if line.startswith('['):
                key = line.split(']')[0][1:]
                value = line.split(']')[1].strip()
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
    return params


def parse_iopad(filepath):
    """
    解析 .iopad 文件
    返回:
        dies: dict, {die_name: {pad_name: (x1, y1, x2, y2)}}
        pad_to_die: dict, {pad_name: die_name}
    """
    dies = {}
    pad_to_die = {}
    current_die = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('[DIE]'):
                current_die = line.split(']')[1].strip()
                dies[current_die] = {}
            else:
                parts = line.split()
                if len(parts) >= 5:
                    pad_name = parts[0]
                    coords = (float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4]))
                    dies[current_die][pad_name] = coords
                    pad_to_die[pad_name] = current_die
    return dies, pad_to_die


def parse_bump(filepath):
    """解析 .bump 文件，返回 bump 列表 [(x1, y1, x2, y2), ...]"""
    bumps = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                bumps.append((float(parts[0]), float(parts[1]),
                              float(parts[2]), float(parts[3])))
    return bumps


def parse_net(filepath):
    """解析 .net 文件，返回 net 列表 [(net_name, pad1, pad2), ...]"""
    nets = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                nets.append((parts[0], parts[1], parts[2]))
    return nets


# ============== 辅助函数 ==============

def get_die_bbox(pads):
    """计算某个 die 所有 IO pad 的包围框"""
    if not pads:
        return None
    xs = []
    ys = []
    for coords in pads.values():
        xs.extend([coords[0], coords[2]])
        ys.extend([coords[1], coords[3]])
    margin_x = (max(xs) - min(xs)) * 0.05 + 50
    margin_y = (max(ys) - min(ys)) * 0.05 + 50
    return (min(xs) - margin_x, min(ys) - margin_y,
            max(xs) + margin_x, max(ys) + margin_y)


def pad_center(coords):
    """返回 pad 的中心坐标"""
    return ((coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2)


# ============== 可视化函数 ==============

def visualize_benchmark(benchmark_dir):
    """可视化一个 benchmark"""

    # 获取 benchmark 名称
    benchmark_name = os.path.basename(os.path.normpath(benchmark_dir))

    # 构建文件路径
    gp_file = os.path.join(benchmark_dir, f"{benchmark_name}.gp")
    iopad_file = os.path.join(benchmark_dir, f"{benchmark_name}.iopad")
    bump_file = os.path.join(benchmark_dir, f"{benchmark_name}.bump")
    net_file = os.path.join(benchmark_dir, f"{benchmark_name}.net")

    # 检查文件
    for f in [gp_file, iopad_file, bump_file, net_file]:
        if not os.path.exists(f):
            print(f"错误: 文件不存在 - {f}")
            return

    # 解析数据
    params = parse_gp(gp_file)
    dies, pad_to_die = parse_iopad(iopad_file)
    bumps = parse_bump(bump_file)
    nets = parse_net(net_file)

    # 收集所有参与 net 的 pad
    net_pads = set()
    for _, pad1, pad2 in nets:
        net_pads.add(pad1)
        net_pads.add(pad2)

    # 统计信息
    total_pads = sum(len(pads) for pads in dies.values())
    print(f"=== Benchmark: {benchmark_name} ===")
    print(f"  封装尺寸: {params.get('PACKAGEWIDTH', '?')} x {params.get('PACKAGEHEIGHT', '?')}")
    print(f"  布线层数: {int(params.get('LAYERNUMBER', 0))}")
    print(f"  线宽: {params.get('WIREWIDTH', '?')}, Via宽: {params.get('VIAWIDTH', '?')}")
    print(f"  Die 数量: {len(dies)}")
    print(f"  IO Pad 总数: {total_pads}")
    print(f"  Bump 总数: {len(bumps)}")
    print(f"  Net 总数: {len(nets)} (chip-to-chip)")
    print(f"  Chip-to-board pad: {total_pads - len(net_pads)}")

    # ---- 创建图形 ----
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Benchmark: {benchmark_name}\n'
                 f'Package: {params.get("PACKAGEWIDTH", "?")}×{params.get("PACKAGEHEIGHT", "?")}  '
                 f'Layers: {int(params.get("LAYERNUMBER", 0))}  '
                 f'Dies: {len(dies)}  Nets: {len(nets)}  Bumps: {len(bumps)}',
                 fontsize=14, fontweight='bold')

    # 颜色方案
    die_colors = plt.cm.Set2(np.linspace(0, 1, max(len(dies), 3)))
    die_color_map = {}
    for i, die_name in enumerate(sorted(dies.keys())):
        die_color_map[die_name] = die_colors[i]

    pkg_w = params.get('PACKAGEWIDTH', 5000)
    pkg_h = params.get('PACKAGEHEIGHT', 5000)

    # ======== 左图: 总览 ========
    ax1 = axes[0]
    ax1.set_title('总览视图 (Overview)', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')

    # 封装边界
    pkg_rect = patches.Rectangle((0, 0), pkg_w, pkg_h,
                                  linewidth=2, edgecolor='black',
                                  facecolor='#f0f0f0', zorder=1)
    ax1.add_patch(pkg_rect)

    # 绘制 bump pad (灰色小点)
    if bumps:
        bump_xs = [(b[0] + b[2]) / 2 for b in bumps]
        bump_ys = [(b[1] + b[3]) / 2 for b in bumps]
        # 计算 bump 大小用于显示
        bump_size = max(bumps[0][2] - bumps[0][0], bumps[0][3] - bumps[0][1])
        marker_size = max(0.5, min(3, 5000 / len(bumps) * 2))
        ax1.scatter(bump_xs, bump_ys, s=marker_size, c='#cccccc',
                    marker='s', zorder=2, label=f'Bumps ({len(bumps)})')

    # 绘制每个 die 的边界框和 IO pad
    legend_handles = []
    for die_name in sorted(dies.keys()):
        pads = dies[die_name]
        color = die_color_map[die_name]
        bbox = get_die_bbox(pads)

        if bbox:
            # Die 边界框
            die_rect = patches.FancyBboxPatch(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0], bbox[3] - bbox[1],
                boxstyle="round,pad=0",
                linewidth=1.5, edgecolor=color, facecolor=(*color[:3], 0.1),
                zorder=3)
            ax1.add_patch(die_rect)

            # Die 名称标签
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            ax1.text(cx, cy, die_name, fontsize=10, fontweight='bold',
                     ha='center', va='center', color=color,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor=color, alpha=0.8),
                     zorder=10)

        # 绘制 IO pad
        for pad_name, coords in pads.items():
            x1, y1, x2, y2 = coords
            w = x2 - x1
            h = y2 - y1
            is_net_pad = pad_name in net_pads
            fc = color if is_net_pad else (*color[:3], 0.3)
            ec = 'black' if is_net_pad else 'gray'
            lw = 0.5

            pad_rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=5)
            ax1.add_patch(pad_rect)

        # 图例
        legend_handles.append(patches.Patch(
            facecolor=color, edgecolor='black',
            label=f'{die_name} ({len(pads)} pads)'))

    # 绘制 net 连接线
    all_pads = {}
    for die_name, pads in dies.items():
        all_pads.update(pads)

    for net_name, pad1, pad2 in nets:
        if pad1 in all_pads and pad2 in all_pads:
            c1 = pad_center(all_pads[pad1])
            c2 = pad_center(all_pads[pad2])
            die1 = pad_to_die.get(pad1, '')
            die2 = pad_to_die.get(pad2, '')
            ax1.plot([c1[0], c2[0]], [c1[1], c2[1]],
                     color='red', linewidth=0.8, alpha=0.5, zorder=4)

    legend_handles.append(Line2D([0], [0], color='red', linewidth=1.5,
                                  alpha=0.6, label=f'Nets ({len(nets)})'))
    legend_handles.append(Line2D([0], [0], marker='s', color='w',
                                  markerfacecolor='#cccccc', markersize=6,
                                  label=f'Bumps ({len(bumps)})'))

    ax1.legend(handles=legend_handles, loc='upper right', fontsize=9,
               framealpha=0.9)
    ax1.set_xlim(-pkg_w * 0.05, pkg_w * 1.05)
    ax1.set_ylim(-pkg_h * 0.05, pkg_h * 1.05)
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.grid(True, alpha=0.2)

    # ======== 右图: Net 详细视图 ========
    ax2 = axes[1]
    ax2.set_title('Net 连接详图 (Net Connections)', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')

    # 封装边界
    pkg_rect2 = patches.Rectangle((0, 0), pkg_w, pkg_h,
                                   linewidth=2, edgecolor='black',
                                   facecolor='#fafafa', zorder=1)
    ax2.add_patch(pkg_rect2)

    # Die 边界框
    for die_name in sorted(dies.keys()):
        pads = dies[die_name]
        color = die_color_map[die_name]
        bbox = get_die_bbox(pads)
        if bbox:
            die_rect = patches.FancyBboxPatch(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0], bbox[3] - bbox[1],
                boxstyle="round,pad=0",
                linewidth=1.5, edgecolor=color, facecolor=(*color[:3], 0.08),
                zorder=2)
            ax2.add_patch(die_rect)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            ax2.text(cx, cy, die_name, fontsize=11, fontweight='bold',
                     ha='center', va='center', color=color,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor=color, alpha=0.8),
                     zorder=10)

    # 只绘制参与 net 的 IO pad，并标注名称
    for die_name in sorted(dies.keys()):
        pads = dies[die_name]
        color = die_color_map[die_name]
        for pad_name, coords in pads.items():
            if pad_name in net_pads:
                x1, y1, x2, y2 = coords
                w = x2 - x1
                h = y2 - y1
                pad_rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=0.8, edgecolor='black', facecolor=color, zorder=5)
                ax2.add_patch(pad_rect)

    # 用不同颜色区分不同 die pair 的 net
    die_pairs = {}
    for net_name, pad1, pad2 in nets:
        d1 = pad_to_die.get(pad1, '?')
        d2 = pad_to_die.get(pad2, '?')
        pair = tuple(sorted([d1, d2]))
        if pair not in die_pairs:
            die_pairs[pair] = []
        die_pairs[pair].append((net_name, pad1, pad2))

    pair_colors = plt.cm.tab10(np.linspace(0, 1, max(len(die_pairs), 1)))
    net_legend = []

    for idx, (pair, pair_nets) in enumerate(sorted(die_pairs.items())):
        pc = pair_colors[idx % len(pair_colors)]
        for net_name, pad1, pad2 in pair_nets:
            if pad1 in all_pads and pad2 in all_pads:
                c1 = pad_center(all_pads[pad1])
                c2 = pad_center(all_pads[pad2])
                ax2.annotate('', xy=c2, xytext=c1,
                             arrowprops=dict(arrowstyle='->', color=pc,
                                             lw=1.2, alpha=0.7),
                             zorder=4)
        net_legend.append(Line2D([0], [0], color=pc, linewidth=2, alpha=0.7,
                                  label=f'{pair[0]}↔{pair[1]} ({len(pair_nets)} nets)'))

    ax2.legend(handles=net_legend, loc='upper right', fontsize=9,
               framealpha=0.9)
    ax2.set_xlim(-pkg_w * 0.05, pkg_w * 1.05)
    ax2.set_ylim(-pkg_h * 0.05, pkg_h * 1.05)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(benchmark_dir, f"{benchmark_name}_visualization.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")

    plt.show()


# ============== 批量可视化 ==============

def visualize_all(benchmarks_root):
    """批量可视化所有 benchmark"""
    dirs = sorted([d for d in os.listdir(benchmarks_root)
                   if os.path.isdir(os.path.join(benchmarks_root, d))])

    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    fig.suptitle('All Benchmarks Overview', fontsize=16, fontweight='bold')

    for idx, dir_name in enumerate(dirs):
        if idx >= 10:
            break
        row = idx // 5
        col = idx % 5
        ax = axes[row][col]

        benchmark_dir = os.path.join(benchmarks_root, dir_name)
        gp_file = os.path.join(benchmark_dir, f"{dir_name}.gp")
        iopad_file = os.path.join(benchmark_dir, f"{dir_name}.iopad")
        bump_file = os.path.join(benchmark_dir, f"{dir_name}.bump")
        net_file = os.path.join(benchmark_dir, f"{dir_name}.net")

        if not all(os.path.exists(f) for f in [gp_file, iopad_file, bump_file, net_file]):
            ax.text(0.5, 0.5, f'{dir_name}\n(missing files)', ha='center', va='center')
            continue

        params = parse_gp(gp_file)
        dies, pad_to_die = parse_iopad(iopad_file)
        bumps = parse_bump(bump_file)
        nets = parse_net(net_file)

        pkg_w = params.get('PACKAGEWIDTH', 5000)
        pkg_h = params.get('PACKAGEHEIGHT', 5000)
        total_pads = sum(len(p) for p in dies.values())

        # 所有 pad
        all_pads = {}
        for die_name, pads in dies.items():
            all_pads.update(pads)

        die_colors = plt.cm.Set2(np.linspace(0, 1, max(len(dies), 3)))

        # 封装边界
        ax.add_patch(patches.Rectangle((0, 0), pkg_w, pkg_h,
                                        linewidth=1.5, edgecolor='black',
                                        facecolor='#f5f5f5', zorder=1))

        # Bumps
        if bumps:
            bx = [(b[0] + b[2]) / 2 for b in bumps]
            by = [(b[1] + b[3]) / 2 for b in bumps]
            ms = max(0.3, min(2, 3000 / len(bumps)))
            ax.scatter(bx, by, s=ms, c='#cccccc', marker='s', zorder=2)

        # Die 和 pad
        for i, die_name in enumerate(sorted(dies.keys())):
            pads = dies[die_name]
            color = die_colors[i]
            bbox = get_die_bbox(pads)
            if bbox:
                ax.add_patch(patches.FancyBboxPatch(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0], bbox[3] - bbox[1],
                    boxstyle="round,pad=0",
                    linewidth=1, edgecolor=color,
                    facecolor=(*color[:3], 0.15), zorder=3))

            for pad_name, coords in pads.items():
                x1, y1, x2, y2 = coords
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=0.3, edgecolor='black',
                    facecolor=color, zorder=5))

        # Nets
        for _, pad1, pad2 in nets:
            if pad1 in all_pads and pad2 in all_pads:
                c1 = pad_center(all_pads[pad1])
                c2 = pad_center(all_pads[pad2])
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                        color='red', linewidth=0.6, alpha=0.5, zorder=4)

        ax.set_xlim(-pkg_w * 0.05, pkg_w * 1.05)
        ax.set_ylim(-pkg_h * 0.05, pkg_h * 1.05)
        ax.set_aspect('equal')
        ax.set_title(f'{dir_name}\n{len(dies)}die  {len(nets)}net  '
                     f'{total_pads}pad  {len(bumps)}bump\n'
                     f'{int(pkg_w)}×{int(pkg_h)}  L{int(params.get("LAYERNUMBER", 0))}',
                     fontsize=9)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    output_path = os.path.join(benchmarks_root, "all_benchmarks_overview.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"总览图已保存到: {output_path}")
    plt.show()


# ============== 主程序 ==============

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  python visualize.py <benchmark_name>   # 可视化单个 benchmark")
        print("  python visualize.py all                 # 可视化所有 benchmark")
        print()
        print("示例:")
        print("  python visualize.py dense2")
        print("  python visualize.py pkg1")
        print("  python visualize.py all")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    arg = sys.argv[1]

    if arg.lower() == 'all':
        visualize_all(script_dir)
    else:
        benchmark_dir = os.path.join(script_dir, arg)
        if not os.path.isdir(benchmark_dir):
            print(f"错误: 找不到 benchmark 目录 - {benchmark_dir}")
            sys.exit(1)
        visualize_benchmark(benchmark_dir)
