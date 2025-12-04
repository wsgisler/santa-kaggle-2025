import math
import itertools
import csv
from decimal import Decimal

import gurobipy as gp
from gurobipy import GRB

from shapely.geometry import Polygon
from shapely import affinity


###############################################################################
# Your ChristmasTree class
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
# Geometry helpers: convex hull + rotation
###############################################################################

def get_base_hull_vertices():
    """
    Use ChristmasTree at angle 0 to build the convex hull polygon.
    Returns hull vertices [(x,y), ...] in CCW order (float) and the hull polygon.
    """
    tree = ChristmasTree(center_x='0', center_y='0', angle='0')
    poly = tree.polygon
    hull = poly.convex_hull
    coords = list(hull.exterior.coords)[:-1]  # drop repeated last point
    return [(float(x), float(y)) for (x, y) in coords], hull


def rotate_point(x, y, angle_deg):
    r = math.radians(angle_deg)
    c = math.cos(r)
    s = math.sin(r)
    return c * x - s * y, s * x + c * y


def build_rotated_hull(base_hull, angle_deg):
    """Rotate base hull vertices by angle_deg."""
    return [rotate_point(x, y, angle_deg) for (x, y) in base_hull]


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


###############################################################################
# Main solver using hull-SAT with correct disjunction
###############################################################################

def pack_trees_fixed_angles_hull(angles_deg, L_UB, eps=0.0):
    """
    angles_deg : list of length N, fixed angle (degrees) for each tree.
    L_UB       : upper bound on square side length.
    eps        : optional positive gap along axes (0 allows touching).
    Returns:
      centers  : list of (cx, cy)
      best_L   : optimal (or best found) square side
    """

    N = len(angles_deg)

    # Base hull from canonical tree
    base_hull, base_poly = get_base_hull_vertices()

    # Rotated hull for each tree
    hulls = [build_rotated_hull(base_hull, ang) for ang in angles_deg]

    # Precompute axes and body projection offsets for each pair
    # axes[(i,j)] = list of dicts with keys:
    #   'ax','ay','min_i','max_i','min_j','max_j'
    axes = {}
    max_body_proj = 0.0

    for i, j in itertools.combinations(range(N), 2):
        h_i = hulls[i]
        h_j = hulls[j]

        # Collect normals from both hulls
        axis_list = []
        axis_list.extend(edge_normals(h_i))
        axis_list.extend(edge_normals(h_j))

        # Optional: deduplicate axes (by direction). For simplicity we skip it;
        # the hull is tiny anyway.

        pair_axes = []

        for (ax, ay) in axis_list:
            # Projection offsets for tree i (body only)
            vals_i = [ax * x + ay * y for (x, y) in h_i]
            min_i = min(vals_i)
            max_i = max(vals_i)

            # Projection offsets for tree j (body only)
            vals_j = [ax * x + ay * y for (x, y) in h_j]
            min_j = min(vals_j)
            max_j = max(vals_j)

            max_body_proj = max(
                max_body_proj,
                abs(min_i), abs(max_i),
                abs(min_j), abs(max_j)
            )

            pair_axes.append(
                {
                    "ax": ax,
                    "ay": ay,
                    "min_i": min_i,
                    "max_i": max_i,
                    "min_j": min_j,
                    "max_j": max_j,
                }
            )

        axes[(i, j)] = pair_axes

    # Big-M on projection inequalities:
    # projection difference between centers is at most 2*sqrt(2)*L_UB,
    # plus body offsets.
    max_center_proj = 2.0 * math.sqrt(2.0) * L_UB
    M = max_center_proj + 2.0 * max_body_proj

    # Build model
    m = gp.Model("christmas_tree_packing_hull_fixed_angles")

    # Square side length (variable)
    L = m.addVar(lb=0.0, ub=L_UB, vtype=GRB.CONTINUOUS, name="L")

    # Tree centers (bounded by [0, L_UB])
    cx = m.addVars(N, lb=0.0, ub=L_UB, vtype=GRB.CONTINUOUS, name="cx")
    cy = m.addVars(N, lb=0.0, ub=L_UB, vtype=GRB.CONTINUOUS, name="cy")

    # Binary variables y_left[(i,j,a)], y_right[(i,j,a)] per pair and axis index
    y_left = {}
    y_right = {}
    for (i, j), ax_list in axes.items():
        for a_idx, _info in enumerate(ax_list):
            y_left[(i, j, a_idx)] = m.addVar(vtype=GRB.BINARY,
                                             name=f"yL_{i}_{j}_{a_idx}")
            y_right[(i, j, a_idx)] = m.addVar(vtype=GRB.BINARY,
                                              name=f"yR_{i}_{j}_{a_idx}")

    m.update()

    # Containment: all hull vertices must lie in [0, L] x [0, L]
    for i in range(N):
        h = hulls[i]
        for (vx, vy) in h:
            m.addConstr(cx[i] + vx >= 0.0)
            m.addConstr(cy[i] + vy >= 0.0)
            m.addConstr(cx[i] + vx <= L)
            m.addConstr(cy[i] + vy <= L)

    # Non-overlap via separating axes on hulls
    for (i, j), ax_list in axes.items():
        # At least one axis (with a direction) separates hull i and hull j
        m.addConstr(
            gp.quicksum(
                y_left[(i, j, a_idx)] + y_right[(i, j, a_idx)]
                for a_idx in range(len(ax_list))
            ) >= 1,
            name=f"pair_sep_{i}_{j}"
        )

        for a_idx, info in enumerate(ax_list):
            ax = info["ax"]
            ay = info["ay"]
            min_i = info["min_i"]
            max_i = info["max_i"]
            min_j = info["min_j"]
            max_j = info["max_j"]

            # Projections:
            # I_i = [ax*cx[i] + ay*cy[i] + min_i, ax*cx[i] + ay*cy[i] + max_i]
            # I_j = [ax*cx[j] + ay*cy[j] + min_j, ax*cx[j] + ay*cy[j] + max_j]

            # i before j along axis a
            m.addConstr(
                ax * cx[i] + ay * cy[i] + max_i
                <= ax * cx[j] + ay * cy[j] + min_j - eps
                   + M * (1 - y_left[(i, j, a_idx)]),
                name=f"sepL_{i}_{j}_{a_idx}"
            )

            # j before i along axis a
            m.addConstr(
                ax * cx[j] + ay * cy[j] + max_j
                <= ax * cx[i] + ay * cy[i] + min_i - eps
                   + M * (1 - y_right[(i, j, a_idx)]),
                name=f"sepR_{i}_{j}_{a_idx}"
            )

    # Objective: minimize L
    m.setObjective(L, GRB.MINIMIZE)

    # Reasonable defaults
    m.Params.MIPGap = 0.01   # 1% optimality gap
    # m.Params.TimeLimit = 3600  # set if desired
    
    m.Params.MIPFocus   = 1
    m.Params.Cuts       = 2
    m.Params.Presolve   = 2
    m.Params.Heuristics = 0.5
    m.Params.MIPGap     = 0.02

    m.optimize()

    # if m.Status not in (GRB.INFEASIBLE, GRB.UNBOUNDED):
#         raise RuntimeError(f"Model ended with status {m.Status}")

    centers = [(cx[i].X, cy[i].X) for i in range(N)]
    best_L = L.X
    return centers, best_L


###############################################################################
# Wrapper: solve and write solution_N_trees.csv
###############################################################################

def solve_and_write_csv(angles_deg, L_UB):
    """
    angles_deg : list of fixed angles (degrees) per tree.
    L_UB       : upper bound on square side.
    Produces: solution_<N>_trees.csv with columns id,x,y,deg
    """
    N = len(angles_deg)
    centers, best_L = pack_trees_fixed_angles_hull(angles_deg, L_UB)

    filename = f"solution_{N}_trees.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])
        for i, ((x, y), deg) in enumerate(zip(centers, angles_deg)):
            # example: id = "004_01" for N=4, first tree
            tree_id = f"{N:03d}_{i+1:02d}"
            writer.writerow([tree_id, 's'+str(x), 's'+str(y), 's'+str(deg)])

    print(f"Best L ≈ {best_L:.6f}")
    print(f"Wrote solution to {filename}")
    return filename, best_L


###############################################################################
# Example main
###############################################################################

if __name__ == "__main__":
    # Try with small N first, e.g. N=2 or N=4
    N = 20

    # Example: both trees upright (0 degrees); change as you like
    angles_deg = [0.0 for _ in range(N)]
    angles_deg = [246.722085493092,155.951579746336,308.755353648396,7.213480884367,54.911647286091,246.005604344723,189.915812750370,354.006877789002,10.321214380826,56.943135555461,246.346960705784,191.061290147400,336.761845824111,10.522188301021,50.610814470300,250.821111382189,204.571039032483,337.399550601912,23.983221537743,66.238991234958]
    angles_deg = angles_deg[:N]

    # Estimate a loose upper bound L_UB
    base_hull, base_poly = get_base_hull_vertices()
    minx, miny, maxx, maxy = base_poly.bounds
    approx_size = max(maxx - minx, maxy - miny)
    L_UB = N * approx_size  # very loose but safe

    solve_and_write_csv(angles_deg, L_UB)
