import shapely
print(f'Using shapely {shapely.__version__}')

import math
import os
import random
from decimal import Decimal, getcontext
from time import time

import random
import copy

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
scale_factor = Decimal('1e15')

from decimal import Decimal
from shapely.geometry import Polygon
from shapely import affinity

scale_factor = Decimal("1.0")   # or whatever you used

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
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                (top_w / 2 * scale_factor, tier_1_y * scale_factor),
                (top_w / 4 * scale_factor, tier_1_y * scale_factor),
                (mid_w / 2 * scale_factor, tier_2_y * scale_factor),
                (mid_w / 4 * scale_factor, tier_2_y * scale_factor),
                (base_w / 2 * scale_factor, base_y * scale_factor),
                (trunk_w / 2 * scale_factor, base_y * scale_factor),
                (trunk_w / 2 * scale_factor, trunk_bottom_y * scale_factor),
                (-trunk_w / 2 * scale_factor, trunk_bottom_y * scale_factor),
                (-trunk_w / 2 * scale_factor, base_y * scale_factor),
                (-base_w / 2 * scale_factor, base_y * scale_factor),
                (-mid_w / 4 * scale_factor, tier_2_y * scale_factor),
                (-mid_w / 2 * scale_factor, tier_2_y * scale_factor),
                (-top_w / 4 * scale_factor, tier_1_y * scale_factor),
                (-top_w / 2 * scale_factor, tier_1_y * scale_factor),
            ]
        )

        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )

    def move_by(self, dx, dy, dangle):
        self.center_x += Decimal(dx)
        self.center_y += Decimal(dy)
        self.angle += Decimal(dangle)
        self._build_polygon()

BIG = 1e6   # large penalty weight

def load_and_normalize_trees(n):
    """Load trees from CSV and normalize into first quadrant; return (trees, side_length)."""
    solution_path = f'solutions/solution_{n}_trees.csv'
    df = pd.read_csv(solution_path)

    trees = []
    for _, row in df.iterrows():
        center_x = row['x']
        center_y = row['y']
        angle = row['deg']
        tree = ChristmasTree(
            center_x=Decimal(center_x.replace('s','')),
            center_y=Decimal(center_y.replace('s','')),
            angle=Decimal(angle.replace('s',''))
        )
        trees.append(tree)

    trees, side_length = normalize_trees(trees)
    return trees, side_length

def total_overlap_area(trees):
    """Sum of pairwise intersection areas."""
    area = 0.0
    n = len(trees)
    for i in range(n):
        for j in range(i + 1, n):
            inter = trees[i].polygon.intersection(trees[j].polygon)
            if not inter.is_empty:
                area += inter.area
    return area

def boundary_violation(trees, L):
    """Penalty for sticking out of the [0, L] x [0, L] square."""
    penalty = 0.0
    for t in trees:
        minx, miny, maxx, maxy = t.polygon.bounds
        if minx < 0:
            penalty += (0 - minx) ** 2
        if miny < 0:
            penalty += (0 - miny) ** 2
        if maxx > L:
            penalty += (maxx - L) ** 2
        if maxy > L:
            penalty += (maxy - L) ** 2
    return penalty

def energy(trees, L):
    return BIG * total_overlap_area(trees) + BIG * boundary_violation(trees, L)*2

def biased_delta(step_xy, shrink_bias):
    """
    Return a delta for one coordinate.
    - coord: current coordinate (Decimal)
    - step_xy: max step size (float)
    - shrink_bias: in [0,1]. 0 = unbiased, 1 = always move toward 0.
    """
    # with probability shrink_bias, choose a direction that reduces |coord|
    if random.random() < shrink_bias:
        return -random.uniform(0, step_xy)
    else:
        # unbiased move
        return random.uniform(-step_xy, step_xy)

def anneal_layout(trees, L,
                  n_steps=20000,
                  T_start=1.0,
                  T_end=1e-3,
                  step_xy=0.05,
                  step_angle=10.0,
                  shrink_bias=0.0):
    """
    Simulated annealing to find a non-overlapping layout inside [0, L]^2.
    Returns (best_trees_copy, best_energy).
    """

    # Work on a copy so we don't destroy the caller's trees
    current = copy.deepcopy(trees)
    best = copy.deepcopy(trees)

    E_current = energy(current, L)
    E_best = E_current

    for k in range(n_steps):
        if E_current < 1e-15:
            print('Perfect solution found, stopping early.')
            break
        if k % 500 == 0:
            print(f'Step {k}/{n_steps}, current energy: {E_current:.6f}, best energy: {E_best:.6f}')
        # Exponential temperature schedule
        t = k / (n_steps - 1)
        T = T_start * (T_end / T_start) ** t

        # Pick a random tree and propose a random move
        idx = random.randrange(len(current))
        tree = current[idx]

        old_state = (tree.center_x, tree.center_y, tree.angle)
        old_polygon = tree.polygon

        dx = biased_delta(step_xy, shrink_bias)
        dy = biased_delta(step_xy, shrink_bias)
        dangle = random.uniform(-step_angle, step_angle)

        tree.move_by(dx, dy, dangle)

        E_new = energy(current, L)
        dE = E_new - E_current

        # Metropolis criterion
        if dE <= 0 or random.random() < math.exp(-dE / T):
            E_current = E_new
            if E_new <= E_best:
                E_best = E_new
                best = copy.deepcopy(current)
        else:
            # Reject move, restore old state
            tree.center_x, tree.center_y, tree.angle = old_state
            tree.polygon = old_polygon

    return best, E_best

def find_min_square(trees,
                    L_start,
                    shrink_factor=0.95,
                    min_L=0.5,
                    **anneal_kwargs):
    """
    Continuously try smaller squares until we fail to find a feasible layout.
    Returns (best_L, best_layout).
    """

    L = L_start
    current_layout = copy.deepcopy(trees)
    best_L = L
    best_layout = copy.deepcopy(trees)

    while L > min_L:
        print('Trying with side length = {:.12f}'.format(L))
        layout, E = anneal_layout(current_layout, L, **anneal_kwargs)
        if E < 1e-3:      # essentially zero overlaps & violations
            # Success — remember this layout and try smaller square
            best_L = L
            best_layout = copy.deepcopy(layout)
            L *= shrink_factor
            current_layout = layout      # start next run from this
        else:
            # Could not pack them into this small square - try retries
            print(f'Failed with L = {L:.12f} (E = {E:.6f}). Attempting retries...')
            success = False
            for retry in range(1, 3):
                L_retry = L * (1.0 + (1 - shrink_factor)/3 * retry)  # Slightly increase L for retry
                print(f'Retry {retry}/2 with slightly larger L = {L_retry:.12f}')
                layout_retry, E_retry = anneal_layout(current_layout, L_retry, **anneal_kwargs)
                if E_retry < 1e-3:
                    print(f'Retry {retry} succeeded! Continuing from L = {L_retry:.12f}')
                    best_L = L_retry
                    best_layout = copy.deepcopy(layout_retry)
                    L = L_retry * shrink_factor
                    current_layout = layout_retry
                    success = True
                    break
                else:
                    print(f'Retry {retry} failed (E = {E_retry:.6f})')
            
            if not success:
                print(f'All retries exhausted. Stopping optimization.')
                break

    return best_L, best_layout

def normalize_trees(trees):
    """Translate trees so that all fit into the first quadrant, and return side length."""
    min_x = min(t.polygon.bounds[0] for t in trees)
    min_y = min(t.polygon.bounds[1] for t in trees)

    for t in trees:
        t.move_by(-min_x, -min_y, 0)

    max_x = max(t.polygon.bounds[2] for t in trees)
    max_y = max(t.polygon.bounds[3] for t in trees)

    side_length = max(max_x, max_y)
    return trees, side_length

def improve_solution(n):
    trees, side_length = load_and_normalize_trees(n)

    best_L, best_layout = find_min_square(
        trees,
        L_start=side_length,
        shrink_factor=0.995,
        n_steps=20000,
        step_xy=0.06,
        step_angle=10.0,
        shrink_bias=0.3
    )

    print("Smallest feasible side_length:", best_L)

    # write to solutions/sa_solution_123_trees.csv
    output_rows = []
    for i, tree in enumerate(best_layout):
        output_rows.append({
            'id': f'{n:03d}_{i}',
            'x': f's{tree.center_x:.12f}',
            'y': f's{tree.center_y:.12f}',
            'deg': f's{tree.angle:.12f}'
        })
    output_df = pd.DataFrame(output_rows)
    output_path = f'solutions/sa_solution_{n}_trees.csv'
    output_df.to_csv(output_path, index=False)
    print(f'Wrote improved solution to {output_path}.')

for i in range(3, 35):
    improve_solution(i)

# from itertools import product

# def tune_sa_parameters(
#     n,
#     param_space,
#     n_trials=1,
#     min_L=0.5
# ):
#     """
#     Tune SA / packing parameters for instance `n`.

#     Parameters
#     ----------
#     n : int
#         Instance id (uses solutions/solution_{n}_trees.csv).
#     param_space : dict
#         Mapping from parameter name to list of values.
#         Valid keys include:
#           - 'shrink_factor' (for find_min_square)
#           - Any kwargs for anneal_layout, e.g.:
#                 'n_steps', 'T_start', 'T_end',
#                 'step_xy', 'step_angle', 'shrink_bias'
#     n_trials : int
#         How many times to repeat each parameter combo (to average randomness).
#     min_L : float
#         Passed to find_min_square.

#     Returns
#     -------
#     best_params : dict
#         Parameter setting with smallest mean best_L.
#     results_df : pandas.DataFrame
#         Table of all tested combinations and their scores.
#     """
#     param_names = list(param_space.keys())
#     combos = list(product(*[param_space[name] for name in param_names]))

#     records = []

#     print(f"Total parameter combinations: {len(combos)}")
#     combo_count = 0

#     for combo in combos:
#         combo_count += 1
#         params = dict(zip(param_names, combo))
#         print(f"\n=== Combo {combo_count}/{len(combos)}: {params}")

#         best_Ls = []
#         times = []

#         for trial in range(n_trials):
#             trees, side_length = load_and_normalize_trees(n)
#             start_time = time()

#             # Separate out shrink_factor (belongs to find_min_square)
#             # vs the rest (go to anneal_layout via **anneal_kwargs)
#             sf = params.get("shrink_factor", 0.995)
#             anneal_kwargs = {k: v for k, v in params.items() if k != "shrink_factor"}

#             best_L, _ = find_min_square(
#                 trees,
#                 L_start=side_length,
#                 shrink_factor=sf,
#                 min_L=min_L,
#                 **anneal_kwargs
#             )

#             elapsed = time() - start_time
#             best_Ls.append(best_L)
#             times.append(elapsed)

#             print(f"   trial {trial+1}/{n_trials}: best_L={best_L:.6f}, time={elapsed:.1f}s")

#         mean_L = float(sum(best_Ls) / len(best_Ls))
#         std_L = float(np.std(best_Ls))
#         mean_time = float(sum(times) / len(times))

#         records.append({
#             **params,
#             "mean_best_L": mean_L,
#             "std_best_L": std_L,
#             "mean_time_s": mean_time,
#         })

#     results_df = pd.DataFrame(records).sort_values("mean_best_L")
#     print("\nTop parameter settings (sorted by mean_best_L):")
#     print(results_df.head())

#     best_row = results_df.iloc[0]
#     best_params = {name: best_row[name] for name in param_names}

#     return best_params, results_df

# # for i in range(75,80):
# #     improve_solution(i)

# param_space = {
#     "shrink_factor": [0.99, 0.995],
#     "n_steps": [10000, 20000],
#     "step_xy": [0.04, 0.06],
#     "step_angle": [5.0, 10.0],
#     "shrink_bias": [0.2, 0.4],
#     # T_start / T_end can also be included if you want:
#     # "T_start": [0.5, 1.0],
#     # "T_end": [1e-3],
# }

# best_params, results_df = tune_sa_parameters(
#     n=75,
#     param_space=param_space,
#     n_trials=2,   # average over 2 runs per combo
#     min_L=0.5
# )

# print("\nBest parameters found:")
# print(best_params)