import shapely
print(f'Using shapely {shapely.__version__}')

import math
import os
import random
from decimal import Decimal, getcontext
from time import time

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

# Build the index of the submission, in the format:
#  <trees_in_problem>_<tree_index>

index = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]

class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))

def find_good_subsolutions(trees, target_size, threads = 4, time_limit=300):
    """For a given placement, this finds the best subsolution with target_size trees (best = smallest bounding square side length)."""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()

    # For each tree, define a binary variable, whether it is placed or not
    x = {t: model.NewBoolVar(f'x_{i}') for i, t in enumerate(trees)}

    # target_size number of trees should be in the solution
    model.Add(sum(x[trees[i]] for i in range(len(trees))) == target_size)

    minx = model.NewIntVar(-int(1e16), int(1e16), 'minx')
    miny = model.NewIntVar(-int(1e16), int(1e16), 'miny')
    maxx = model.NewIntVar(-int(1e16), int(1e16), 'maxx')
    maxy = model.NewIntVar(-int(1e16), int(1e16), 'maxy')
    for t in trees:
        # If tree t is placed, its bounds contribute to minx, miny, maxx, maxy
        model.Add(minx <= int(t.polygon.bounds[0])).OnlyEnforceIf(x[t])
        model.Add(miny <= int(t.polygon.bounds[1])).OnlyEnforceIf(x[t])
        model.Add(maxx >= int(t.polygon.bounds[2])).OnlyEnforceIf(x[t])
        model.Add(maxy >= int(t.polygon.bounds[3])).OnlyEnforceIf(x[t])

    side_length = model.NewIntVar(0, int(1e17), 'side_length')
    model.AddMaxEquality(side_length, [
        maxx - minx,
        maxy - miny
    ])

    # Objective: Minimize the side length of the bounding square
    model.Minimize(side_length)
    #model.Minimize(0)  # Dummy objective since we cannot express the bounding square easily

    solver = cp_model.CpSolver()
    # Enable logging
    # solver.parameters.log_search_progress = True
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
        return placed_trees, side_length
    else:
        print('No solution found.')
    
def plot_results(side_length, placed_trees, num_trees):
    """Plots the arrangement of trees and the bounding square."""
    _, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis([i / num_trees for i in range(num_trees)])

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    for i, tree in enumerate(placed_trees):
        # Rescale for plotting
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

    padding = 0.5
    ax.set_xlim(
        float(square_x - Decimal(str(padding))),
        float(square_x + side_length + Decimal(str(padding))))
    ax.set_ylim(float(square_y - Decimal(str(padding))),
                float(square_y + side_length + Decimal(str(padding))))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    score = side_length*side_length/num_trees
    plt.title(f'{num_trees} Trees: {side_length:.12f}, score: {score:.12f}')
    plt.show()
    plt.close()
    
for i in range(190,201):
    # import the solution solutions/solution_{i}_trees.csv
    solution_path = f'solutions/solution_{i}_trees.csv'
    df = pd.read_csv(solution_path)
    trees = []
    for _, row in df.iterrows():
        tree_id = row['id']
        center_x = row['x']
        center_y = row['y']
        angle = row['deg']
        tree = ChristmasTree(center_x=Decimal(center_x.replace('s','')), center_y=Decimal(center_y.replace('s','')), angle=Decimal(angle.replace('s','')))
        bounds = tree.polygon.bounds
        trees.append(tree)
    for target_size in range(2,i):
        current_placed_trees, side_length = find_good_subsolutions(trees, target_size)
        score = side_length*side_length/target_size
        n = len(current_placed_trees)
        print(f'From solution with {i} trees, found subsolution with {target_size} trees, side length {side_length:.12f}, score {score:.12f}')
        # write the solution to a csv file. The format is id,x,y,deg
        output_rows = []
        for ii, tree in enumerate(current_placed_trees):
            output_rows.append({
                'id': f'{target_size:03d}_{ii}',
                'x': f's{tree.center_x:.12f}',
                'y': f's{tree.center_y:.12f}',
                'deg': f's{tree.angle:.12f}'
            })
        output_df = pd.DataFrame(output_rows)
        # write to solutions/solution_{n}_trees.csv if not exists or if the score is better than the existing one
        output_path = f'solutions/solution_{n}_trees.csv'
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            existing_side = Decimal('0')
            # compute existing side length from existing_df
            existing_trees = []
            for _, row in existing_df.iterrows():
                x = Decimal(row['x'][1:]) # Remove leading 's'
                y = Decimal(row['y'][1:])  # Remove leading 's'
                deg = Decimal(row['deg'][1:])  # Remove leading 's'
                tree = ChristmasTree(center_x=x, center_y=y, angle=deg)
                existing_trees.append(tree)
            all_polygons = [t.polygon for t in existing_trees]
            bounds = unary_union(all_polygons).bounds
            minx = Decimal(bounds[0]) / scale_factor
            miny = Decimal(bounds[1]) / scale_factor
            maxx = Decimal(bounds[2]) / scale_factor
            maxy = Decimal(bounds[3]) / scale_factor
            width = maxx - minx
            height = maxy - miny
            existing_side = Decimal(str(max(width, height)))
            existing_score = existing_side*existing_side/Decimal(str(n))
            if score < existing_score:
                print(f'New solution for n = {n} is better (score {score:.12f} < existing score {existing_score:.12f}), overwriting.')
                output_df.to_csv(output_path, index=False)
            else:
                print(f'Existing solution for n = {n} is better (score {existing_score:.12f} <= new score {score:.12f}), not overwriting.')
        else:
            print(f'Trees: {n}, New score: {score}, No existing solution, saving new one.')
            output_df.to_csv(output_path, index=False)