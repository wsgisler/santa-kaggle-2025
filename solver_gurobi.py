import math
import itertools
import csv
from decimal import Decimal

import gurobipy as gp
from gurobipy import GRB

from shapely.geometry import Polygon
from shapely.ops import triangulate
from shapely import affinity


###############################################################################
# Your ChristmasTree class (unchanged)
###############################################################################

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


###############################################################################
# Geometry helpers
###############################################################################

def rotate_point(x, y, angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return c * x - s * y, s * x + c * y


def edge_normals(poly):
    """
    Given a convex polygon as a list of vertices [(x1,y1),...],
    return a list of *unit* outward normals for each edge.
    """
    normals = []
    m = len(poly)
    for k in range(m):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % m]
        ex, ey = x2 - x1, y2 - y1
        nx, ny = -ey, ex
        L = math.hypot(nx, ny)
        if L > 1e-12:
            normals.append((nx / L, ny / L))
    return normals


def build_rotated_parts(base_parts, angle_deg):
    """
    base_parts: list of convex polygons, each a list of (x,y),
                all centered/positioned in body coordinates.
    angle_deg: fixed rotation for this tree.
    returns: list of rotated convex polygons (still in body coordinates).
    """
    angle_rad = math.radians(angle_deg)
    rot_parts = []
    for poly in base_parts:
        rot_poly = [rotate_point(x, y, angle_rad) for (x, y) in poly]
        rot_parts.append(rot_poly)
    return rot_parts


###############################################################################
# Build canonical tree parts (convex) using ChristmasTree + Shapely triangulate
###############################################################################

def build_base_tree_parts():
    # Canonical tree at angle 0, at origin:
    tree = ChristmasTree(center_x='0', center_y='0', angle='0')
    poly = tree.polygon  # shapely Polygon

    # Triangulate the non-convex tree polygon into convex triangles
    triangles = triangulate(poly)

    base_parts = []
    for tri in triangles:
        # tri.exterior.coords gives closed ring; drop last point = first point
        coords = list(tri.exterior.coords)[:-1]
        base_parts.append([(float(x), float(y)) for (x, y) in coords])

    return base_parts, poly


###############################################################################
# Main solver: exact non-overlap with fixed angles, minimize square side
###############################################################################

def pack_trees_with_fixed_angles(base_parts, angles_deg, L_UB, eps=1e-4):
    """
    base_parts : list of convex polygons (triangles) for canonical tree,
                 each polygon is [(x1,y1), ...].
    angles_deg : list of length N with fixed angle (degrees) for each tree.
    L_UB       : upper bound on square side length.
    eps        : separation gap along a separating axis.
    """
    N = len(angles_deg)
    print(f"\n{'='*70}")
    print(f"Building model for {N} trees")
    print(f"{'='*70}")

    # Pre-rotate polygons for each tree according to its fixed angle
    print(f"\n[1/7] Pre-rotating tree parts for each tree...")
    tree_parts = []      # tree_parts[i] = list of convex polygons for tree i
    for i in range(N):
        tree_parts.append(build_rotated_parts(base_parts, angles_deg[i]))
        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"  Rotated {i + 1}/{N} trees")

    # Precompute candidate axes and bounds for big-M
    print(f"\n[2/7] Precomputing separating axes for tree pairs...")
    axes = {}  # axes[(i,j)] = list of (ax, ay) candidate directions
    max_body_proj = 0.0
    
    total_pairs = N * (N - 1) // 2
    pair_count = 0

    for i, j in itertools.combinations(range(N), 2):
        pair_count += 1
        if pair_count % 100 == 0 or pair_count == total_pairs:
            print(f"  Processed {pair_count}/{total_pairs} tree pairs")
        ax_list = []
        # normals from all parts of i and j
        for t in (i, j):
            for part in tree_parts[t]:
                ax_list.extend(edge_normals(part))

        axes[(i, j)] = ax_list

        # track max projection of body vertices for big-M
        for t in (i, j):
            for part in tree_parts[t]:
                for (ax, ay) in ax_list:
                    vals = [ax * vx + ay * vy for (vx, vy) in part]
                    mn = min(vals)
                    mx = max(vals)
                    max_body_proj = max(max_body_proj, abs(mn), abs(mx))

    # Big-M for projection separation:
    # projection = ax*cx + ay*cy + body_offset
    # ax,ay are unit; cx,cy in [0,L_UB] => |ax*cx+ay*cy| <= sqrt(2)*L_UB
    max_positional_proj = math.sqrt(2.0) * L_UB
    M = 2.0 * (max_positional_proj + max_body_proj)
    print(f"  Computed Big-M value: {M:.2f}")
    
    total_axes = sum(len(ax_list) for ax_list in axes.values())
    print(f"  Total separating axes: {total_axes}")

    # Build model
    print(f"\n[3/7] Creating Gurobi model...")
    m = gp.Model("christmas_tree_packing_fixed_angles")

    # Square side length (decision variable) with known upper bound
    L = m.addVar(lb=0.0, ub=L_UB, vtype=GRB.CONTINUOUS, name="L")
    print(f"  Added square side length variable (L_UB={L_UB:.2f})")

    # Tree centers
    cx = m.addVars(N, lb=0.0, ub=L_UB, vtype=GRB.CONTINUOUS, name="cx")
    cy = m.addVars(N, lb=0.0, ub=L_UB, vtype=GRB.CONTINUOUS, name="cy")
    print(f"  Added {2*N} center position variables")

    # Projection interval bounds per (tree, part, pair, axis)
    print(f"\n[4/7] Adding projection and binary variables...")
    Pi_min = {}
    Pi_max = {}
    z = {}  # binary axis selector per pair and axis
    
    var_count = 0
    binary_count = 0

    for (i, j), ax_list in axes.items():
        n_axes = len(ax_list)

        # binary for each axis
        for a_idx in range(n_axes):
            z[(i, j, a_idx)] = m.addVar(vtype=GRB.BINARY,
                                        name=f"z_{i}_{j}_{a_idx}")
            binary_count += 1

        # projection bounds for each tree in the pair and each of its parts
        for t in (i, j):
            for p_idx, part in enumerate(tree_parts[t]):
                for a_idx in range(n_axes):
                    Pi_min[(t, p_idx, i, j, a_idx)] = m.addVar(
                        lb=-GRB.INFINITY,
                        vtype=GRB.CONTINUOUS,
                        name=f"Pmin_{t}_{p_idx}_{i}_{j}_{a_idx}"
                    )
                    Pi_max[(t, p_idx, i, j, a_idx)] = m.addVar(
                        lb=-GRB.INFINITY,
                        vtype=GRB.CONTINUOUS,
                        name=f"Pmax_{t}_{p_idx}_{i}_{j}_{a_idx}"
                    )
                    var_count += 2

    m.update()
    print(f"  Added {binary_count} binary variables (axis selectors)")
    print(f"  Added {var_count} continuous variables (projection bounds)")
    print(f"  Total variables: {m.NumVars}")

    # Containment constraints: all vertices must be within [0, L] x [0, L]
    print(f"\n[5/7] Adding containment constraints...")
    containment_count = 0
    for i in range(N):
        for part in tree_parts[i]:
            for (vx, vy) in part:
                X = cx[i] + vx
                Y = cy[i] + vy
                m.addConstr(X >= 0.0)
                m.addConstr(Y >= 0.0)
                m.addConstr(X <= L)
                m.addConstr(Y <= L)
                containment_count += 4
        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"  Added constraints for {i + 1}/{N} trees")
    print(f"  Total containment constraints: {containment_count}")

    # Projection constraints
    print(f"\n[6/7] Adding projection constraints...")
    projection_count = 0
    pair_idx = 0
    for (i, j), ax_list in axes.items():
        pair_idx += 1
        poly_i_parts = tree_parts[i]
        poly_j_parts = tree_parts[j]

        for a_idx, (ax, ay) in enumerate(ax_list):

            # tree i
            for p_idx, part in enumerate(poly_i_parts):
                for (vx, vy) in part:
                    X = cx[i] + vx
                    Y = cy[i] + vy
                    proj = ax * X + ay * Y
                    m.addConstr(Pi_min[(i, p_idx, i, j, a_idx)] <= proj)
                    m.addConstr(Pi_max[(i, p_idx, i, j, a_idx)] >= proj)

            # tree j
            for p_idx, part in enumerate(poly_j_parts):
                for (vx, vy) in part:
                    X = cx[j] + vx
                    Y = cy[j] + vy
                    proj = ax * X + ay * Y
                    m.addConstr(Pi_min[(j, p_idx, i, j, a_idx)] <= proj)
                    m.addConstr(Pi_max[(j, p_idx, i, j, a_idx)] >= proj)
                    projection_count += 2
        
        if pair_idx % 100 == 0 or pair_idx == total_pairs:
            print(f"  Processed {pair_idx}/{total_pairs} pairs")
    print(f"  Total projection constraints: {projection_count}")

    # Non-overlap via separating axes
    print(f"\n[7/7] Adding non-overlap (separation) constraints...")
    separation_count = 0
    pair_idx = 0
    for (i, j), ax_list in axes.items():
        pair_idx += 1
        n_axes = len(ax_list)

        # at least one axis separates polygons i and j
        m.addConstr(
            gp.quicksum(z[(i, j, a_idx)] for a_idx in range(n_axes)) >= 1,
            name=f"pair_sep_{i}_{j}"
        )
        separation_count += 1

        for a_idx in range(n_axes):
            for p_i, part_i in enumerate(tree_parts[i]):
                for p_j, part_j in enumerate(tree_parts[j]):
                    # i-part vs j-part separation on axis a_idx
                    m.addConstr(
                        Pi_max[(i, p_i, i, j, a_idx)]
                        <= Pi_min[(j, p_j, i, j, a_idx)] - eps
                           + M * (1 - z[(i, j, a_idx)]),
                        name=f"sep_ij_{i}_{p_i}_{j}_{p_j}_{a_idx}_1"
                    )
                    m.addConstr(
                        Pi_max[(j, p_j, i, j, a_idx)]
                        <= Pi_min[(i, p_i, i, j, a_idx)] - eps
                           + M * (1 - z[(i, j, a_idx)]),
                        name=f"sep_ij_{i}_{p_i}_{j}_{p_j}_{a_idx}_2"
                    )
                    separation_count += 2
        
        if pair_idx % 100 == 0 or pair_idx == total_pairs:
            print(f"  Processed {pair_idx}/{total_pairs} pairs")
    print(f"  Total separation constraints: {separation_count}")

    # Model summary
    print(f"\n{'='*70}")
    print(f"MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"Variables: {m.NumVars} ({binary_count} binary, {m.NumVars - binary_count} continuous)")
    print(f"Constraints: {containment_count + projection_count + separation_count}")
    print(f"  - Containment: {containment_count}")
    print(f"  - Projection: {projection_count}")
    print(f"  - Separation: {separation_count}")
    print(f"{'='*70}\n")

    # Objective: minimize square side L
    m.setObjective(L, GRB.MINIMIZE)

    # Some reasonable defaults – tune as you like
    m.Params.MIPGap = 0.01      # 1% optimality gap
    # m.Params.TimeLimit = 3600  # uncomment to limit runtime in seconds
    # log output = True
    m.Params.OutputFlag = 1

    print("Starting optimization...\n")
    m.optimize()
    print("\nOptimization completed!")

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Model ended with status {m.Status}")

    centers = [(cx[i].X, cy[i].X) for i in range(N)]
    best_L = L.X
    return centers, best_L


###############################################################################
# Wrapper: solves and writes solution_N_trees.csv
###############################################################################

def solve_and_write_csv(angles_deg, L_UB):
    """
    angles_deg : list of fixed angles (degrees) per tree.
    L_UB       : upper bound on square side.
    """
    N = len(angles_deg)

    # Build base tree geometry from your class
    base_parts, base_poly = build_base_tree_parts()

    centers, L_val = pack_trees_with_fixed_angles(
        base_parts=base_parts,
        angles_deg=angles_deg,
        L_UB=L_UB
    )

    filename = f"solution_{N}_trees.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])
        for i, ((x, y), deg) in enumerate(zip(centers, angles_deg)):
            tree_id = f"{N:03d}_{i+1:02d}"   # e.g. 004_01
            writer.writerow([tree_id, x, y, deg])

    print(f"Best L ≈ {L_val:.6f}")
    print(f"Wrote solution to {filename}")
    return filename, L_val


###############################################################################
# Example main: 178 trees with some fixed angles
###############################################################################

if __name__ == "__main__":
    # ---- Set your angles per tree here ----
    N = 24

    # Example: alternating 0 and 180 degrees.
    # Replace this with your actual angle pattern.
    angles_deg = [156.712714595229,285.370408247760,465.117367580036,471.737045692802,92.100400246971,471.737045692802,66.443087307366,144.975532666338,177.695530111234,337.295251535049,66.443087307366,120.564447854462,-508.485404902967,-74.925656023450,92.100400246971,156.712714595229,177.695530111234,-74.925656023450,74.865584404995,246.408916684964,27.335180417678,156.712714595229,337.295251535049,246.408916684964,285.370408247760,28.846980050788,27.335180417678,74.865584404995,428.385872422967,177.695530111234,74.865584404995,428.385872422967,28.846980050788,428.385872422967,27.335180417678,144.975532666338,-74.925656023450,28.846980050788,156.712714595229,28.846980050788,120.564447854462,465.117367580036,246.408916684964,74.865584404995,92.100400246971,471.737045692802,337.295251535049,177.695530111234,-508.485404902967,465.117367580036,-74.925656023450,66.443087307366,120.564447854462,471.737045692802,144.975532666338,-508.485404902967,246.408916684964,92.100400246971,-508.485404902967,66.443087307366,285.370408247760,428.385872422967,27.335180417678,337.295251535049,120.564447854462,465.117367580036,144.975532666338]
    angles_deg = angles_deg[:N]  # truncate or extend as needed

    # ---- Choose an upper bound for the square side ----
    # Use the base polygon bounds to get a reasonable width/height:
    base_parts, base_poly = build_base_tree_parts()
    minx, miny, maxx, maxy = base_poly.bounds
    approx_size = max(maxx - minx, maxy - miny)
    L_UB = N * approx_size  # very loose but safe; tighten if you can

    # Re-use base_parts from above (to avoid recomputing triangulation)
    # so we pass them to the solver via a small shim:
    def run():
        centers, L_val = pack_trees_with_fixed_angles(
            base_parts=base_parts,
            angles_deg=angles_deg,
            L_UB=L_UB
        )
        filename = f"solution_{N}_trees.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "deg"])
            for i, ((x, y), deg) in enumerate(zip(centers, angles_deg)):
                tree_id = f"{N:03d}_{i+1:02d}"
                writer.writerow([tree_id, x, y, deg])
        print(f"Best L ≈ {L_val:.6f}")
        print(f"Wrote solution to {filename}")

    run()
