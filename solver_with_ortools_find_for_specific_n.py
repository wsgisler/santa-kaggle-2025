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

def solve_with_ortools(sizemax = 3, num_samples = 1000, threads = 8, time_limit = 300):
    # first generate num_samples random trees with random positions and angles. The max x position and y position is sizemax
    trees = []
    num_samp = 0
    while num_samp < num_samples:
        x = random.uniform(-0.2, sizemax)
        y = random.uniform(-0.2, sizemax)
        angle = random.uniform(0, 360)
        tree = ChristmasTree(center_x=x, center_y=y, angle=angle)
        # check if the tree is within the area of sizemax x sizemax
        minx, miny, maxx, maxy = tree.polygon.bounds
        if minx >= 0 and miny >= 0 and maxx <= sizemax * float(scale_factor) and maxy <= sizemax * float(scale_factor):
            num_samp += 1
            trees.append(tree)

    """Solves the Christmas tree placement problem using OR-Tools."""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()

    # For each tree, define a binary variable, whether it is placed or not
    x = {t: model.NewBoolVar(f'x_{i}') for i, t in enumerate(trees)}

    # Add non-overlapping constraints
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            tree_i = trees[i]
            tree_j = trees[j]
            if tree_i.polygon.intersects(tree_j.polygon) and not touches(tree_i.polygon, tree_j.polygon):
                model.AddBoolOr([x[tree_i].Not(), x[tree_j].Not()])

    # Objective: Maximize the number of placed trees
    model.Maximize(sum(x[trees[i]] for i in range(len(trees))))

    solver = cp_model.CpSolver()
    # Enable logging
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
    
sm = 3.3
while sm <= 11:
    sm += 0.01
    current_placed_trees, side = solve_with_ortools(sizemax=sm, num_samples = 700, time_limit =120, threads=8)
    n = len(current_placed_trees)
    score = side*side/n
    #plot_results(side, current_placed_trees, n)
    # write the solution to a csv file. The format is id,x,y,deg
    output_rows = []
    for i, tree in enumerate(current_placed_trees):
        output_rows.append({
            'id': f'{n:03d}_{i}',
            'x': f's{tree.center_x:.12f}',
            'y': f's{tree.center_y:.12f}',
            'deg': f's{tree.angle:.12f}'
        })
    output_df = pd.DataFrame(output_rows)
    # write to solutions/ortools_solution_{n}_trees.csv if not exists or if the score is better than the existing one
    output_path = f'solutions/ortools_solution_{n}_trees.csv'
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
        existing_score = existing_side*existing_side/n
        print('Current sizemax:', sm, 'Trees:', n, 'New score:', score, 'Existing score:', existing_score)
        if score < existing_score:
            print(f'New solution is better (score {score:.12f} < existing score {existing_score:.12f}), overwriting.')
            output_df.to_csv(output_path, index=False)
        else:
            print(f'Existing solution is better (score {existing_score:.12f} <= new score {score:.12f}), not overwriting.')
    else:
        print('Current sizemax:', sm, 'Trees:', n, 'New score:', score, 'No existing solution, saving new one.')
        output_df.to_csv(output_path, index=False)