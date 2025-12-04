# combine all solutions from the folder "solutions/" into a single csv file: submission.csv
import os
import pandas as pd
from decimal import Decimal, getcontext
scale_factor = Decimal('1e15')
from shapely.geometry import Polygon
from shapely import affinity

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
        
def read_solution_file(filepath):
    df = pd.read_csv(filepath)
    trees = []
    for _, row in df.iterrows():
        tree_id = row['id']
        center_x = row['x']
        center_y = row['y']
        angle = row['deg']
        tree = ChristmasTree(center_x=Decimal(center_x.replace('s','')), center_y=Decimal(center_y.replace('s','')), angle=Decimal(angle.replace('s','')))
        bounds = tree.polygon.bounds
        trees.append(tree)
    return trees

def evaluate_solution(trees):
    minx = Decimal('inf')
    miny = Decimal('inf')
    maxx = Decimal('-inf')
    maxy = Decimal('-inf')
    
    for tree in trees:
        bounds = tree.polygon.bounds
        minx = min(Decimal(bounds[0]) / scale_factor, minx)
        miny = min(Decimal(bounds[1]) / scale_factor, miny)
        maxx = max(Decimal(bounds[2]) / scale_factor, maxx)
        maxy = max(Decimal(bounds[3]) / scale_factor, maxy)
    
    # check whether there are overlaps:
    overlaps = False
    for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            if trees[i].polygon.intersects(trees[j].polygon):
                print(f'Overlap detected between tree {i} and tree {j}!')
                overlaps = True
    
    side_length = max(maxx - minx, maxy - miny)
    score = side_length*side_length/len(trees)
    return side_length, score, overlaps


def integrate_solutions():
    """
    Reads all CSV files from to_integrate_solutions/ and its subfolders.
    Compares each sa_solution_{n}_trees.csv with solution_{n}_trees.csv in solutions/.
    Replaces the existing solution if the new one has a lower score.
    """
    import re
    import shutil
    
    replacements_count = 0
    to_integrate_dir = 'to_integrate_solutions'
    solutions_dir = 'solutions'
    
    # Pattern to match: sa_solution_{n}_trees.csv
    pattern = re.compile(r'^solution_(\d+)_trees\.csv$')
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(to_integrate_dir):
        for filename in files:
            match = pattern.match(filename)
            if match:
                n = match.group(1)
                new_solution_path = os.path.join(root, filename)
                existing_solution_path = os.path.join(solutions_dir, f'solution_{n}_trees.csv')
                
                print(f'\n--- Processing: {filename} (n={n}) ---')
                
                # Check if existing solution exists
                if not os.path.exists(existing_solution_path):
                    print(f'  No existing solution found at {existing_solution_path}. Skipping.')
                    continue
                
                # Read and evaluate new solution
                try:
                    new_trees = read_solution_file(new_solution_path)
                    new_side_length, new_score, new_overlaps = evaluate_solution(new_trees)
                    print(f'  New solution: side_length={new_side_length:.12f}, score={new_score:.12f}, overlaps={new_overlaps}')
                except Exception as e:
                    print(f'  Error reading new solution: {e}')
                    continue
                
                # Read and evaluate existing solution
                try:
                    existing_trees = read_solution_file(existing_solution_path)
                    existing_side_length, existing_score, existing_overlaps = evaluate_solution(existing_trees)
                    print(f'  Existing solution: side_length={existing_side_length:.12f}, score={existing_score:.12f}, overlaps={existing_overlaps}')
                except Exception as e:
                    print(f'  Error reading existing solution: {e}')
                    continue
                
                # Compare scores (lower is better)
                if new_score < existing_score:
                    print(f'  ✓ New solution is better! Replacing...')
                    shutil.copy2(new_solution_path, existing_solution_path)
                    replacements_count += 1
                else:
                    print(f'  ✗ Existing solution is better or equal. Keeping existing.')
    
    print(f'\n=== Integration Complete ===')
    print(f'Total solutions replaced: {replacements_count}')
    return replacements_count


if __name__ == '__main__':
    integrate_solutions()

