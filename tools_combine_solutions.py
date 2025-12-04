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

f = open('stats.csv','w')
f.write('filename,num_trees,side_length,score,overlaps\n')
total_score = 0
pd.set_option('display.float_format', '{:.12f}'.format)
solution_folder = 'solutions/'
submission_rows = []
for filename in os.listdir(solution_folder):
    minx=100
    maxx=-100
    miny=100
    maxy=-100
    trees = []
    if filename.endswith('.csv'):
        filepath = os.path.join(solution_folder, filename)
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            tree_id = row['id']
            center_x = row['x']
            center_y = row['y']
            angle = row['deg']
            tree = ChristmasTree(center_x=Decimal(center_x.replace('s','')), center_y=Decimal(center_y.replace('s','')), angle=Decimal(angle.replace('s','')))
            bounds = tree.polygon.bounds
            trees.append(tree)
            minx = min(Decimal(bounds[0]) / scale_factor, minx)
            miny = min(Decimal(bounds[1]) / scale_factor, miny)
            maxx = max(Decimal(bounds[2]) / scale_factor, maxx)
            maxy = max(Decimal(bounds[3]) / scale_factor, maxy)
            submission_rows.append({
                'id': tree_id,
                'x': center_x,
                'y': center_y,
                'deg': angle
            })
        # check whether there are overlaps:
        overlaps = 'No'
        for i in range(len(trees)):
            for j in range(i+1, len(trees)):
                if trees[i].polygon.intersects(trees[j].polygon):
                    print(f'Overlap detected in file {filename} between tree {i} and tree {j}!')
                    overlaps = 'Yes'
        side_length = max(maxx - minx, maxy - miny)
        score = side_length*side_length/len(trees)
        total_score += score
        f.write(f'{filename},{len(trees)},{side_length:.12f},{score:.12f},{overlaps}\n')
f.close()
submission_df = pd.DataFrame(submission_rows)
submission_df = submission_df.sort_values(by=['id'])
submission_df.to_csv('submission.csv', index=False)
print(f'Wrote submission.csv with {len(submission_df)} trees.')
print(f'Total score: {total_score:.12f}')