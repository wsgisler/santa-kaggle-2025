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

def find_comb_solution(trees, multiplicator=2, leave_away=0):
    """ For a given placement, this finds new solutions by combining solutions in a grid pattern."""
    # for example, if the multiplicator is 2, every tree in the original solution is repeated 4 times in a 2x2 grid, if 3 then 9 times in a 3x3 grid, etc.
    # The distance between the trees is determined by the bounding box of the original solution.
    new_trees = []
    all_polygons = [t.polygon for t in trees]
    bounds = unary_union(all_polygons).bounds
    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor
    width = maxx - minx
    height = maxy - miny
    x_offset = width #+ Decimal('0.000000000001')  # add small gap between trees
    y_offset = height #+ Decimal('0.000000000001')  # add small gap between trees
    for i in range(multiplicator):
        for j in range(multiplicator):
            for t in trees:
                new_center_x = t.center_x + Decimal(str(i)) * x_offset
                new_center_y = t.center_y + Decimal(str(j)) * y_offset
                new_tree = ChristmasTree(center_x=new_center_x, center_y=new_center_y, angle=t.angle)
                new_trees.append(new_tree)

    if leave_away > 0:
        # randomly leave away leave_away trees
        random.shuffle(new_trees)
        new_trees = new_trees[:-leave_away]
    
    return new_trees, max(width * Decimal(str(multiplicator)), height * Decimal(str(multiplicator)))
    
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
    
for i in range(2,51):
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
    for multiplicator in range(2,7):
        for leave_away in range(0,5): # if leave away is 1, we leave away one random tree from the combined solution, if 2 then two trees, etc.
            target_size = i * multiplicator*multiplicator-leave_away
            if target_size > 200:
                continue
            current_placed_trees, side_length = find_comb_solution(trees, multiplicator = multiplicator, leave_away = leave_away)
            n = len(current_placed_trees)
            target_size = n
            score = side_length*side_length/target_size
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


# Second loop: Optimize by removing one tree at a time from solutions 200 down to 2
print('\n=== Starting tree removal optimization ===\n')
replacements_made = 0

for n in range(200, 1, -1):
    print(f'\n--- Processing solution with {n} trees ---')
    
    # Read the solution with n trees
    solution_path = f'solutions/solution_{n}_trees.csv'
    if not os.path.exists(solution_path):
        print(f'Solution with {n} trees does not exist, skipping.')
        continue
    
    df = pd.read_csv(solution_path)
    trees = []
    for _, row in df.iterrows():
        tree_id = row['id']
        center_x = row['x']
        center_y = row['y']
        angle = row['deg']
        tree = ChristmasTree(center_x=Decimal(center_x.replace('s','')), center_y=Decimal(center_y.replace('s','')), angle=Decimal(angle.replace('s','')))
        trees.append(tree)
    
    # Read the target solution with (n-1) trees to compare against
    target_n = n - 1
    target_path = f'solutions/solution_{target_n}_trees.csv'
    
    if not os.path.exists(target_path):
        print(f'Target solution with {target_n} trees does not exist, skipping.')
        continue
    
    # Compute existing score for target solution
    existing_df = pd.read_csv(target_path)
    existing_trees = []
    for _, row in existing_df.iterrows():
        x = Decimal(row['x'][1:])  # Remove leading 's'
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
    existing_side = max(width, height)
    existing_score = existing_side * existing_side / Decimal(str(target_n))
    
    print(f'Existing {target_n}-tree solution score: {existing_score:.12f}')
    
    # Try removing each tree one by one
    best_score = existing_score
    best_trees = None
    best_removed_idx = None
    
    for remove_idx in range(len(trees)):
        # Create modified solution without tree at remove_idx
        modified_trees = trees[:remove_idx] + trees[remove_idx+1:]
        
        # Compute bounding box and score
        all_polygons = [t.polygon for t in modified_trees]
        bounds = unary_union(all_polygons).bounds
        minx = Decimal(bounds[0]) / scale_factor
        miny = Decimal(bounds[1]) / scale_factor
        maxx = Decimal(bounds[2]) / scale_factor
        maxy = Decimal(bounds[3]) / scale_factor
        width = maxx - minx
        height = maxy - miny
        side_length = max(width, height)
        score = side_length * side_length / Decimal(str(target_n))
        
        if score < best_score:
            print(f'  Removing tree {remove_idx}: score improved to {score:.12f}')
            best_score = score
            best_trees = modified_trees
            best_removed_idx = remove_idx
    
    # If we found a better solution, save it
    if best_trees is not None:
        print(f'✓ Found better {target_n}-tree solution by removing tree {best_removed_idx} from {n}-tree solution')
        print(f'  New score: {best_score:.12f} < Old score: {existing_score:.12f}')
        
        # Write the improved solution
        output_rows = []
        for ii, tree in enumerate(best_trees):
            output_rows.append({
                'id': f'{target_n:03d}_{ii}',
                'x': f's{tree.center_x:.12f}',
                'y': f's{tree.center_y:.12f}',
                'deg': f's{tree.angle:.12f}'
            })
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(target_path, index=False)
        replacements_made += 1
        print(f'  Saved improved solution to {target_path}')
    else:
        print(f'✗ No improvement found for {target_n}-tree solution')

print(f'\n=== Tree removal optimization complete ===')
print(f'Total solutions replaced: {replacements_made}')
