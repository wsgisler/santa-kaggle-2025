import shapely
print(f'Using shapely {shapely.__version__}')

import math
import os
import random
from decimal import Decimal, getcontext
from time import time

import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from shapely import affinity, touches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

pd.set_option('display.float_format', '{:.12f}'.format)

# Set precision for Decimal
getcontext().prec = 25

scale_factor = Decimal("1.0")


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self._build_polygon()

    def _build_polygon(self):
        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w  = Decimal('0.7')
        mid_w   = Decimal('0.4')
        top_w   = Decimal('0.25')
        tip_y   = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y  = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                (Decimal('0.0'), tip_y),
                (top_w / 2, tier_1_y),
                (top_w / 4, tier_1_y),
                (mid_w / 2, tier_2_y),
                (mid_w / 4, tier_2_y),
                (base_w / 2, base_y),
                (trunk_w / 2, base_y),
                (trunk_w / 2, trunk_bottom_y),
                (-trunk_w / 2, trunk_bottom_y),
                (-trunk_w / 2, base_y),
                (-base_w / 2, base_y),
                (-mid_w / 4, tier_2_y),
                (-mid_w / 2, tier_2_y),
                (-top_w / 4, tier_1_y),
                (-top_w / 2, tier_1_y),
            ]
        )

        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x),
            yoff=float(self.center_y),
        )

    def move_by(self, dx, dy, dangle):
        self.center_x += Decimal(dx)
        self.center_y += Decimal(dy)
        self.angle += Decimal(dangle)
        self._build_polygon()


def build_repeating_pattern(
    t1x, t1y, t1a,
    t2x, t2y, t2a,
    row_x_offset, row_y_offset,
    col_x_offset, col_y_offset,
    n_rows, n_cols
):
    """Create trees for a repeating 2-tree unit cell."""
    trees = []

    base_t1x = Decimal(str(t1x))
    base_t1y = Decimal(str(t1y))
    base_t2x = Decimal(str(t2x))
    base_t2y = Decimal(str(t2y))

    row_dx = Decimal(str(row_x_offset))
    row_dy = Decimal(str(row_y_offset))
    col_dx = Decimal(str(col_x_offset))
    col_dy = Decimal(str(col_y_offset))

    for r in range(n_rows):
        for c in range(n_cols):
            shift_x = c * col_dx + r * row_dx
            shift_y = c * col_dy + r * row_dy

            t1 = ChristmasTree(
                center_x=str(base_t1x + shift_x),
                center_y=str(base_t1y + shift_y),
                angle=str(t1a),
            )
            trees.append(t1)

            t2 = ChristmasTree(
                center_x=str(base_t2x + shift_x),
                center_y=str(base_t2y + shift_y),
                angle=str(t2a),
            )
            trees.append(t2)

    return trees


def check_overlaps(trees):
    polys = [t.polygon for t in trees]
    index = STRtree(polys)
    overlaps = []

    for i, poly in enumerate(polys):
        for j in index.query(poly):
            if j <= i:
                continue
            if poly.intersects(polys[j]) and not poly.touches(polys[j]):
                overlaps.append((i, j))

    return overlaps


def compute_overlap_stats(trees):
    """Return (overlap_area, overlap_count) over all tree pairs."""
    polys = [t.polygon for t in trees]
    index = STRtree(polys)
    overlap_area = 0.0
    overlap_count = 0

    for i, poly in enumerate(polys):
        for j in index.query(poly):
            if j <= i:
                continue
            inter = poly.intersection(polys[j])
            if not inter.is_empty:
                area = inter.area
                if area > 0.0:
                    overlap_area += area
                    overlap_count += 1

    return overlap_area, overlap_count


def compute_side_length(trees):
    """Side length of the bounding square covering all trees."""
    all_polygons = [t.polygon for t in trees]
    minx, miny, maxx, maxy = unary_union(all_polygons).bounds
    width = maxx - minx
    height = maxy - miny
    return max(width, height)


def fmt_number(x):
    s = f"{float(x):.9f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return "s" + s


def write_csv(trees, output_path):
    total = len(trees)
    prefix = f"{total:03d}"

    rows = []
    for idx, t in enumerate(trees):
        rows.append({
            "id": f"{prefix}_{idx}",
            "x": fmt_number(t.center_x),
            "y": fmt_number(t.center_y),
            "deg": f"s{float(t.angle)}",
        })

    df = pd.DataFrame(rows, columns=["id", "x", "y", "deg"])
    df.to_csv(output_path, index=False)


# ------------------------------------------------------------
# 🎄 PLOTTING SUPPORT
# ------------------------------------------------------------
def plot_results(side_length, placed_trees, num_trees):
    """Plots the arrangement of trees and the bounding square."""
    _, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis([i / num_trees for i in range(num_trees)])

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    for i, tree in enumerate(placed_trees):
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(val) / scale_factor for val in x_scaled]
        y = [Decimal(val) / scale_factor for val in y_scaled]
        ax.plot(x, y, color=colors[i])
        ax.fill(x, y, alpha=0.5, color=colors[i])

    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor

    width = maxx - minx
    height = maxy - miny

    side_length = max(width, height)

    square_x = minx if width >= height else minx - (side_length - width) / 2
    square_y = miny if height >= width else miny - (side_length - height) / 2

    bounding_square = Rectangle(
        (float(square_x), float(square_y)),
        float(side_length),
        float(side_length),
        fill=False,
        edgecolor='red',
        linewidth=2,
        linestyle='--',
    )
    ax.add_patch(bounding_square)

    padding = Decimal('0.5')
    ax.set_xlim(float(square_x - padding), float(square_x + side_length + padding))
    ax.set_ylim(float(square_y - padding), float(square_y + side_length + padding))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    score = side_length * side_length / num_trees
    plt.title(f'{num_trees} Trees: {side_length:.12f}, score: {score:.12f}')
    plt.show()
    plt.close()


# ------------------------------------------------------------
# 🔧 FIXED PARAM PARSER
# ------------------------------------------------------------
def parse_maybe_fixed(val):
    """
    Parses values like '3.14' or 'k3.14'.
    Returns (float_value, is_fixed_flag).
    """
    s = str(val)
    if s.startswith("k") or s.startswith("K"):
        return float(s[1:]), True
    return float(s), False


# ------------------------------------------------------------
# 🔥 SIMULATED ANNEALING
# ------------------------------------------------------------
def evaluate_vector(vec, t1x, t1y, n_rows, n_cols, overlap_weight):
    """Compute cost, side-length, overlap stats for a param vector."""
    (
        t1a,
        t2x, t2y, t2a,
        row_x_offset, row_y_offset,
        col_x_offset, col_y_offset,
    ) = vec

    trees = build_repeating_pattern(
        t1x=t1x, t1y=t1y, t1a=t1a,
        t2x=t2x, t2y=t2y, t2a=t2a,
        row_x_offset=row_x_offset, row_y_offset=row_y_offset,
        col_x_offset=col_x_offset, col_y_offset=col_y_offset,
        n_rows=n_rows, n_cols=n_cols,
    )

    overlap_area, overlap_count = compute_overlap_stats(trees)
    side_length = compute_side_length(trees)

    #plot_results(side_length, trees, len(trees ))

    cost = overlap_weight * overlap_area + side_length
    return cost, side_length, overlap_area, overlap_count, trees


def propose_neighbor(vec, step_scales, fixed_flags):
    """
    Randomly perturb one parameter, skipping those marked fixed.
    """
    new_vec = list(vec)

    # indices of free parameters
    free_indices = [i for i, fixed in enumerate(fixed_flags) if not fixed]
    if not free_indices:
        return new_vec  # nothing to change

    idx = random.choice(free_indices)
    step = random.gauss(0.0, step_scales[idx])
    new_vec[idx] += step

    # wrap angles, clamp offsets to a reasonable range
    if idx in (0, 3):  # t1a, t2a
        new_vec[idx] = new_vec[idx] % 360.0
    else:
        # clamp positions/offsets to [-5, 5] to avoid crazy explosions
        new_vec[idx] = max(-5.0, min(5.0, new_vec[idx]))

    return new_vec


def simulated_annealing_optimize(
    initial_vec,
    t1x, t1y,
    n_rows, n_cols,
    fixed_flags,
    max_iter=2000,
    start_temp=1.0,
    end_temp=1e-3,
    overlap_weight=1000.0,
    log_interval=100,
    random_seed=None,
):
    if random_seed is not None:
        random.seed(random_seed)

    current_vec = list(initial_vec)
    (
        current_cost,
        current_side,
        current_overlap_area,
        current_overlap_cnt,
        _,
    ) = evaluate_vector(
        current_vec, t1x, t1y, n_rows, n_cols, overlap_weight
    )

    best_vec = list(current_vec)
    best_cost = current_cost

    best_non_overlap_vec = None
    best_non_overlap_cost = float("inf")
    best_non_overlap_side = None
    best_non_overlap_overlap_area = None

    # If initial solution is non-overlapping, record it
    if current_overlap_area == 0.0:
        best_non_overlap_vec = list(current_vec)
        best_non_overlap_cost = current_cost
        best_non_overlap_side = current_side
        best_non_overlap_overlap_area = current_overlap_area

    # Step scales roughly tuned to tree size
    step_scales = [
        10.0,  # t1a
        0.1,   # t2x
        0.1,   # t2y
        10.0,  # t2a
        0.1,   # row_x_offset
        0.1,   # row_y_offset
        0.1,   # col_x_offset
        0.1,   # col_y_offset
    ]

    for it in range(max_iter):
        # geometric cooling
        t_frac = it / max_iter
        T = start_temp * (end_temp / start_temp) ** t_frac
        if T < 1e-12:
            T = 1e-12

        cand_vec = propose_neighbor(current_vec, step_scales, fixed_flags)
        (
            cand_cost,
            cand_side,
            cand_overlap_area,
            cand_overlap_cnt,
            _,
        ) = evaluate_vector(
            cand_vec, t1x, t1y, n_rows, n_cols, overlap_weight
        )

        delta = cand_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_vec = cand_vec
            current_cost = cand_cost
            current_side = cand_side
            current_overlap_area = cand_overlap_area
            current_overlap_cnt = cand_overlap_cnt

            if current_cost < best_cost:
                best_vec = list(current_vec)
                best_cost = current_cost

            if current_overlap_area == 0.0 and current_cost < best_non_overlap_cost:
                best_non_overlap_vec = list(current_vec)
                best_non_overlap_cost = current_cost
                best_non_overlap_side = current_side
                best_non_overlap_overlap_area = current_overlap_area

        # logging
        if log_interval and it % log_interval == 0:
            best_side_str = (
                f"{best_non_overlap_side:.6f}"
                if best_non_overlap_side is not None
                else "N/A"
            )
            print(
                f"[SA] iter={it:5d} T={T:.4g} "
                f"current_side={current_side:.6f} "
                f"overlap_pairs={current_overlap_cnt} "
                f"overlap_area={current_overlap_area:.6f} "
                f"best_non_overlap_side={best_side_str}"
            )

    return {
        "best_vec": best_vec,
        "best_cost": best_cost,
        "best_non_overlap_vec": best_non_overlap_vec,
        "best_non_overlap_cost": best_non_overlap_cost,
        "best_non_overlap_side": best_non_overlap_side,
        "best_non_overlap_overlap_area": best_non_overlap_overlap_area,
    }


def fallback_non_overlapping(t1x, t1y, n_rows, n_cols):
    """Very loose grid to guarantee a non-overlapping solution."""
    t1a = 0.0
    t2x, t2y, t2a = 3.0, 0.0, 0.0
    row_x_offset, row_y_offset = 0.0, 4.0
    col_x_offset, col_y_offset = 4.0, 0.0

    vec = [
        t1a,
        t2x, t2y, t2a,
        row_x_offset, row_y_offset,
        col_x_offset, col_y_offset,
    ]
    return vec


# ------------------------------------------------------------
# 🚀 MAIN PROGRAM
# ------------------------------------------------------------
def read_existing_side_length(filepath):
    """Read the side length of an existing solution from CSV."""
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath)
        trees = []
        for _, row in df.iterrows():
            x_str = row['x'].replace('s', '') if isinstance(row['x'], str) else str(row['x'])
            y_str = row['y'].replace('s', '') if isinstance(row['y'], str) else str(row['y'])
            deg_str = row['deg'].replace('s', '') if isinstance(row['deg'], str) else str(row['deg'])
            
            tree = ChristmasTree(
                center_x=x_str,
                center_y=y_str,
                angle=deg_str
            )
            trees.append(tree)
        
        return compute_side_length(trees)
    except Exception as e:
        print(f"Warning: Could not read existing solution {filepath}: {e}")
        return None


def should_save_solution(output_path, new_side_length):
    """Check if we should save the new solution."""
    existing_side_length = read_existing_side_length(output_path)
    
    if existing_side_length is None:
        print(f"  No existing solution found. Will save.")
        return True
    
    if new_side_length < existing_side_length:
        print(f"  New solution is better! (side_length: {new_side_length:.6f} < {existing_side_length:.6f})")
        return True
    else:
        print(f"  Existing solution is better. (side_length: {existing_side_length:.6f} <= {new_side_length:.6f})")
        return False


def run_single_configuration(args, rows, cols, output_path):
    """Run optimization for a single rows/cols configuration.
    
    Returns (side_length, trees) tuple if successful, None otherwise.
    """
    t1x = args.t1x
    t1y = args.t1y

    # Parse possibly-fixed parameters
    t1a, fix_t1a = parse_maybe_fixed(args.t1a)
    t2x, fix_t2x = parse_maybe_fixed(args.t2x)
    t2y, fix_t2y = parse_maybe_fixed(args.t2y)
    t2a, fix_t2a = parse_maybe_fixed(args.t2a)

    row_x_offset, fix_row_x = parse_maybe_fixed(args.row_x_offset)
    row_y_offset, fix_row_y = parse_maybe_fixed(args.row_y_offset)
    col_x_offset, fix_col_x = parse_maybe_fixed(args.col_x_offset)
    col_y_offset, fix_col_y = parse_maybe_fixed(args.col_y_offset)

    # Fixed flags for the 8 SA parameters in canonical order
    fixed_flags = [
        fix_t1a,
        fix_t2x, fix_t2y, fix_t2a,
        fix_row_x, fix_row_y,
        fix_col_x, fix_col_y,
    ]

    if args.optimize:
        initial_vec = [
            t1a,
            t2x, t2y, t2a,
            row_x_offset, row_y_offset,
            col_x_offset, col_y_offset,
        ]

        sa_result = simulated_annealing_optimize(
            initial_vec=initial_vec,
            t1x=t1x, t1y=t1y,
            n_rows=rows, n_cols=cols,
            fixed_flags=fixed_flags,
            max_iter=args.sa_iters,
            start_temp=args.sa_start_temp,
            end_temp=args.sa_end_temp,
            overlap_weight=args.overlap_weight,
            log_interval=args.sa_log_interval if args.sa_log_interval else 0,
        )

        if sa_result["best_non_overlap_vec"] is not None:
            vec = sa_result["best_non_overlap_vec"]
            print("  Found non-overlapping solution via SA.")
        else:
            print("  SA did not find a non-overlapping solution; using loose fallback layout.")
            vec = fallback_non_overlapping(t1x, t1y, rows, cols)

        (
            t1a,
            t2x, t2y, t2a,
            row_x_offset, row_y_offset,
            col_x_offset, col_y_offset,
        ) = vec

    # Build pattern with (possibly optimized) parameters
    trees = build_repeating_pattern(
        t1x=t1x, t1y=t1y, t1a=t1a,
        t2x=t2x, t2y=t2y, t2a=t2a,
        row_x_offset=row_x_offset,
        row_y_offset=row_y_offset,
        col_x_offset=col_x_offset,
        col_y_offset=col_y_offset,
        n_rows=rows,
        n_cols=cols,
    )

    overlaps = check_overlaps(trees)
    if overlaps:
        print(f"  WARNING: final arrangement has {len(overlaps)} overlapping pairs!")
        return None
    else:
        print(f"  Final arrangement has no overlaps.")

    side_length = compute_side_length(trees)
    return (side_length, trees)


def main():
    parser = argparse.ArgumentParser(
        description="Generate repeating Christmas-tree patterns, optionally optimized with simulated annealing."
    )

    parser.add_argument("--t1x", type=float, required=True)
    parser.add_argument("--t1y", type=float, required=True)

    # these can be "k..." so we take them as strings and parse later
    parser.add_argument("--t1a", type=str, required=True)
    parser.add_argument("--t2x", type=str, required=True)
    parser.add_argument("--t2y", type=str, required=True)
    parser.add_argument("--t2a", type=str, required=True)

    parser.add_argument("--row_x_offset", type=str, required=True)
    parser.add_argument("--row_y_offset", type=str, required=True)
    parser.add_argument("--col_x_offset", type=str, required=True)
    parser.add_argument("--col_y_offset", type=str, required=True)

    parser.add_argument("--rows", type=int, required=False)
    parser.add_argument("--cols", type=int, required=False)

    # Auto-loop configuration options
    parser.add_argument("--auto_loop", action="store_true",
                        help="Automatically try different rows/cols configurations.")
    parser.add_argument("--config_file", type=str, default=None,
                        help="File with rows,cols pairs (one per line) for auto_loop mode.")
    parser.add_argument("--max_configs", type=int, default=20,
                        help="Maximum number of configurations to try in auto_loop mode.")

    parser.add_argument("--output", type=str, default="pattern.csv")
    parser.add_argument("--plot", action="store_true", help="Enable plotting")

    # Simulated annealing options
    parser.add_argument("--optimize", action="store_true",
                        help="Use simulated annealing to optimize parameters.")
    parser.add_argument("--sa_iters", type=int, default=2000,
                        help="Number of SA iterations.")
    parser.add_argument("--sa_start_temp", type=float, default=1.0)
    parser.add_argument("--sa_end_temp", type=float, default=1e-3)
    parser.add_argument("--overlap_weight", type=float, default=1000.0,
                        help="Penalty weight for overlap area.")
    parser.add_argument("--sa_log_interval", type=int, default=100,
                        help="Log every N SA iterations (0 disables logging).")

    args = parser.parse_args()

    # Auto-loop mode: try different configurations
    if args.auto_loop:
        # Generate or load configurations
        if args.config_file and os.path.exists(args.config_file):
            print(f"Loading configurations from {args.config_file}")
            configs = []
            with open(args.config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) == 2:
                            configs.append((int(parts[0]), int(parts[1])))
        else:
            # Generate a variety of configurations
            print(f"Generating up to {args.max_configs} configurations automatically")
            configs = []
            for rows in range(1, 11):
                for cols in range(1, 11):
                    configs.append((rows, cols))
                    if len(configs) >= args.max_configs:
                        break
                if len(configs) >= args.max_configs:
                    break
        
        print(f"Will try {len(configs)} different configurations")
        
        # Create solutions directory if it doesn't exist
        os.makedirs("solutions", exist_ok=True)
        
        for idx, (rows, cols) in enumerate(configs):
            n_trees = 2 * rows * cols
            output_path = f"solutions/solution_{n_trees}_trees.csv"
            
            print(f"\n{'='*60}")
            print(f"Configuration {idx+1}/{len(configs)}: rows={rows}, cols={cols}, n_trees={n_trees}")
            print(f"{'='*60}")
            
            # Run optimization for this configuration
            result = run_single_configuration(
                args, rows, cols, output_path
            )
            
            if result is not None:
                side_length, trees = result
                
                # Check if we should save this solution
                if should_save_solution(output_path, side_length):
                    write_csv(trees, output_path)
                    print(f"✓ Saved solution to '{output_path}' (side_length: {side_length:.6f})")
                else:
                    print(f"✗ Skipped saving (existing solution is better)")
        
        print(f"\n{'='*60}")
        print("Auto-loop completed!")
        print(f"{'='*60}")
        return
    
    # Single configuration mode (original behavior)
    if args.rows is None or args.cols is None:
        print("Error: --rows and --cols are required when not using --auto_loop mode")
        return

    t1x = args.t1x
    t1y = args.t1y

    # Parse possibly-fixed parameters
    t1a, fix_t1a = parse_maybe_fixed(args.t1a)
    t2x, fix_t2x = parse_maybe_fixed(args.t2x)
    t2y, fix_t2y = parse_maybe_fixed(args.t2y)
    t2a, fix_t2a = parse_maybe_fixed(args.t2a)

    row_x_offset, fix_row_x = parse_maybe_fixed(args.row_x_offset)
    row_y_offset, fix_row_y = parse_maybe_fixed(args.row_y_offset)
    col_x_offset, fix_col_x = parse_maybe_fixed(args.col_x_offset)
    col_y_offset, fix_col_y = parse_maybe_fixed(args.col_y_offset)

    # Store back into args as floats
    args.t1a = t1a
    args.t2x = t2x
    args.t2y = t2y
    args.t2a = t2a
    args.row_x_offset = row_x_offset
    args.row_y_offset = row_y_offset
    args.col_x_offset = col_x_offset
    args.col_y_offset = col_y_offset

    # Fixed flags for the 8 SA parameters in canonical order
    fixed_flags = [
        fix_t1a,
        fix_t2x, fix_t2y, fix_t2a,
        fix_row_x, fix_row_y,
        fix_col_x, fix_col_y,
    ]

    if args.optimize:
        initial_vec = [
            args.t1a,
            args.t2x, args.t2y, args.t2a,
            args.row_x_offset, args.row_y_offset,
            args.col_x_offset, args.col_y_offset,
        ]

        print("Running simulated annealing optimization...")
        sa_result = simulated_annealing_optimize(
            initial_vec=initial_vec,
            t1x=t1x, t1y=t1y,
            n_rows=args.rows, n_cols=args.cols,
            fixed_flags=fixed_flags,
            max_iter=args.sa_iters,
            start_temp=args.sa_start_temp,
            end_temp=args.sa_end_temp,
            overlap_weight=args.overlap_weight,
            log_interval=args.sa_log_interval,
        )

        if sa_result["best_non_overlap_vec"] is not None:
            vec = sa_result["best_non_overlap_vec"]
            print("Found non-overlapping solution via SA.")
        else:
            print("SA did not find a non-overlapping solution; using loose fallback layout.")
            vec = fallback_non_overlapping(t1x, t1y, args.rows, args.cols)

        (
            args.t1a,
            args.t2x, args.t2y, args.t2a,
            args.row_x_offset, args.row_y_offset,
            args.col_x_offset, args.col_y_offset,
        ) = vec

        print("Final optimized parameters:")
        print(f"  t1a={args.t1a:.6f}")
        print(f"  t2x={args.t2x:.6f}, t2y={args.t2y:.6f}, t2a={args.t2a:.6f}")
        print(f"  row_x_offset={args.row_x_offset:.6f}, row_y_offset={args.row_y_offset:.6f}")
        print(f"  col_x_offset={args.col_x_offset:.6f}, col_y_offset={args.col_y_offset:.6f}")

    # Build pattern with (possibly optimized) parameters
    trees = build_repeating_pattern(
        t1x=args.t1x, t1y=args.t1y, t1a=args.t1a,
        t2x=args.t2x, t2y=args.t2y, t2a=args.t2a,
        row_x_offset=args.row_x_offset,
        row_y_offset=args.row_y_offset,
        col_x_offset=args.col_x_offset,
        col_y_offset=args.col_y_offset,
        n_rows=args.rows,
        n_cols=args.cols,
    )

    overlaps = check_overlaps(trees)
    if overlaps:
        # This should only happen if even the fallback somehow overlaps.
        print(f"WARNING: final arrangement still has {len(overlaps)} overlapping pairs!")
    else:
        print("Final arrangement has no overlaps.")

    write_csv(trees, args.output)
    print(f"Wrote {len(trees)} trees to '{args.output}'.")

    if args.plot:
        side_length = compute_side_length(trees)
        plot_results(side_length, trees, len(trees))


if __name__ == "__main__":
    main()
