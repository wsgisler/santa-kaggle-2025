import shapely
print(f'Using shapely {shapely.__version__}')

import math
import os
import random
from decimal import Decimal, getcontext
from time import time

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
scale_factor = Decimal('1')

# Build the index of the submission, in the format:
#  <trees_in_problem>_<tree_index>

index = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]

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

def biased_delta(current_coord, max_l, step_xy, shrink_bias):
    """
    Return a delta for one coordinate.
    - coord: current coordinate (Decimal)
    - step_xy: max step size (float)
    - shrink_bias: in [0,1]. 0 = unbiased, 1 = always move toward 0.
    """
    # with probability shrink_bias, choose a direction that reduces |coord|
    if random.random() < shrink_bias:
        if current_coord > max_l/4*3:
            return -random.uniform(0, step_xy)
        # elif current_coord < max_l/4:
        #     return random.uniform(0, step_xy)
        else:
            return random.uniform(-step_xy, step_xy)
    else:
        # unbiased move
        return random.uniform(-step_xy, step_xy)

def solve_with_ortools(trees, max_L, target_size, num_copies = 2, neighborhood_modifiers = [(0.1, 6, 0)], threads = 8, time_limit = 300, allow_overlaps = False):
    # first generate num_samples random trees with random positions and angles. The max x position and y position is sizemax
    # for each tree, add a copy of the tree with a small change
    sa_trees = []
    for i in range(num_copies):
        modifier = random.choice(neighborhood_modifiers)
        step_xy, step_angle, shrink_bias = modifier
        sa_trees_1 = copy.deepcopy(trees)
        for tree in sa_trees_1:
            dx = biased_delta(tree.center_x, max_L, step_xy, shrink_bias)
            dy = biased_delta(tree.center_y, max_L, step_xy, shrink_bias)
            dangle = random.uniform(-step_angle, step_angle)
            tree.move_by(dx, dy, dangle)
        sa_trees += sa_trees_1

    trees = trees + sa_trees
    
    # remove trees that are not within the target size
    trees = [t for t in trees if t.polygon.bounds[0] >= 0 and t.polygon.bounds[1] >= 0 and Decimal(str(t.polygon.bounds[2]))/scale_factor <= Decimal(str(max_L)) and Decimal(str(t.polygon.bounds[3]))/scale_factor <= Decimal(str(max_L))]

    """Solves the Christmas tree placement problem using OR-Tools."""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    objective = 0

    # For each tree, define a binary variable, whether it is placed or not
    x = {t: model.NewBoolVar(f'x_{i}') for i, t in enumerate(trees)}

    # Add non-overlapping constraints
    if allow_overlaps:
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                # find to what percentage the area of two trees overlap:
                tree_i = trees[i]
                tree_j = trees[j]
                intersection = tree_i.polygon.intersection(tree_j.polygon)
                if not intersection.is_empty:
                    overlap_area = Decimal(str(intersection.area)) / (Decimal(str(tree_i.polygon.area)) + Decimal(str(tree_j.polygon.area)) - Decimal(str(intersection.area)))
                    if overlap_area > 0.1:
                        model.AddBoolOr([x[tree_i].Not(), x[tree_j].Not()])
                    else:
                        # increase objective by 1:
                        helper = model.NewBoolVar(f'helper_{i}_{j}')
                        model.AddBoolOr([x[tree_i].Not(), x[tree_j].Not()]).OnlyEnforceIf(helper.Not())
                        objective += helper
    else:
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                tree_i = trees[i]
                tree_j = trees[j]
                if tree_i.polygon.intersects(tree_j.polygon) and not touches(tree_i.polygon, tree_j.polygon):
                    model.AddBoolOr([x[tree_i].Not(), x[tree_j].Not()])

    # Make sure that the number of placed trees equals target size
    model.Add(sum(x[t] for t in trees) == target_size)

    # Add an objective, but only if overlaps are allowed
    if allow_overlaps:
        model.Minimize(objective)

    solver = cp_model.CpSolver()
    # Enable logging
    if allow_overlaps:
        solver.parameters.log_search_progress = True
    # set parameter for threads and time limt
    solver.parameters.num_search_workers = threads
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    minx = 100
    miny = 100
    maxx = -100
    maxy = -100
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Solution found!')
        # Extract solution and return tree placements
        placed_trees = []
        for i, tree in enumerate(trees):
            if solver.BooleanValue(x[tree]):
                placed_trees.append(tree)
                minx = min(minx, float(tree.polygon.bounds[0])/float(scale_factor))
                miny = min(miny, float(tree.polygon.bounds[1])/float(scale_factor))
                maxx = max(maxx, float(tree.polygon.bounds[2])/float(scale_factor))
                maxy = max(maxy, float(tree.polygon.bounds[3])/float(scale_factor))
        side_length = Decimal(str(max(maxx - minx, maxy - miny)))
        num_overlaps = 0
        if allow_overlaps:
            for i in range(len(placed_trees)):
                for j in range(i + 1, len(placed_trees)):
                    tree_i = placed_trees[i]
                    tree_j = placed_trees[j]
                    if tree_i.polygon.intersects(tree_j.polygon) and not touches(tree_i.polygon, tree_j.polygon):
                        num_overlaps += 1
            print('Number of overlaps in solution:', num_overlaps)
        return placed_trees, num_overlaps
    else:
        print('No solution found.')
        return [], 1

def find_min_square(trees,
                    L_start,
                    shrink_factor=0.95,
                    regrow_factor=1.05,
                    max_num_regrows=5,
                    min_L=0.5,
                    max_failed_tries = 20,
                    num_copies = 2,
                    neighborhood_modifiers = [(0.05, 2, 0.7), 
                                              (0.08, 5, 0.3), 
                                              (0.12, 6, 0.3), 
                                              (0.18, 20, 0), 
                                              (0.4, 30, 0)],
                    threads = 8,
                    time_limit = 300,
                    allow_overlaps_to_escape_local_minima = False):
    """
    Continuously try smaller squares until we fail to find a feasible layout.
    Returns (best_L, best_layout).
    """

    L = L_start
    current_layout = copy.deepcopy(trees)
    best_L = L
    best_layout = copy.deepcopy(trees)

    failed_tries = 0
    last_failed = False
    regrows = 0

    while L > min_L and failed_tries <= max_failed_tries and regrows < max_num_regrows:
        print('Trying with side length = {:.12f}'.format(L))
        if allow_overlaps_to_escape_local_minima:   
            layout, E = solve_with_ortools(current_layout, L, len(current_layout), num_copies=num_copies, neighborhood_modifiers = neighborhood_modifiers, threads=threads, time_limit=time_limit, allow_overlaps = last_failed)
        else:
            layout, E = solve_with_ortools(current_layout, L, len(current_layout), num_copies=num_copies, neighborhood_modifiers = neighborhood_modifiers, threads=threads, time_limit=time_limit, allow_overlaps = False)
        if E < 1e-3:  # essentially zero overlaps & violations
            # Success — remember this layout and try smaller square
            if L < best_L:
                best_L = L
                best_layout = copy.deepcopy(layout)
            L *= shrink_factor
            current_layout = layout      # start next run from this
            failed_tries = 0
            last_failed = False
        else:
           failed_tries += 1
           if len(layout) > 0:
               current_layout = layout
           last_failed = True
           print('Failed tries:', failed_tries, 'Overlaps: ', E)
           if failed_tries == max_failed_tries:
               regrows += 1
               L *= regrow_factor
               print('Regrowing square to {:.12f}, regrow count: {}'.format(L, regrows))
               failed_tries = 0

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

def improve_solution(n,num_copies=10):
    trees, side_length = load_and_normalize_trees(n)

    best_L, best_layout = find_min_square(
        trees,
        L_start=side_length,
        shrink_factor=0.9995,
        max_failed_tries = 2000,
        regrow_factor = 1.004, #1.004
        max_num_regrows=10,
        num_copies = num_copies,
        # neighborhoods: sets of tuples (step_xy, step_angle, shrink_bias)
        neighborhood_modifiers = [(0.03, 1.5, 0), # micro moves
                                  (0.05, 2, 0.7), # tiny moves
                                  (0.08, 5, 0.3),
                                  (0.12, 6, 0.3),
                                  (0.18, 20, 0), # big moves
                                  (0.4, 30, 0)], # big ass moves
        threads = 8,
        time_limit = 300,
        allow_overlaps_to_escape_local_minima=False,
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
    print('Initial side length:', side_length, 'New side length:', best_L)
    print('Initial score: ', side_length*side_length/n, 'New score:', best_L*best_L/n)

# for i in range(8,50):
#     print('------------------ Working with',i,'christmas trees ------------------')
#     num_copies = 700//i
#     improve_solution(i, num_copies=20)

improve_solution(101, num_copies=15)