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
        shrink_factor=0.9998,
        max_failed_tries = 2000,
        regrow_factor = 1.02, #1.004
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

def generate_solution_pool(n, pool_size=20, num_copies=10):
    """Generate a pool of diverse solutions by running improve_solution multiple times."""
    print(f'\n=== Generating solution pool of {pool_size} solutions for {n} trees ===\n')
    
    solution_pool = []
    
    for i in range(pool_size):
        print(f'\n--- Generating solution {i+1}/{pool_size} ---')
        trees, side_length = load_and_normalize_trees(n)
        
        # Use different parameters for diversity
        shrink_factor = 0.998
        
        best_L, best_layout = find_min_square(
            trees,
            L_start=side_length,
            shrink_factor=shrink_factor,
            max_failed_tries=30,
            regrow_factor=1.0001,
            max_num_regrows=2,
            num_copies=num_copies,
            neighborhood_modifiers=[(0.03, 1.5, 0),
                                    (0.05, 2, 0.7),
                                    (0.08, 5, 0.3),
                                    (0.12, 6, 0.3),
                                    (0.18, 20, 0),
                                    (0.4, 30, 0)],
            threads=8,
            time_limit=60,  # Shorter time for pool generation
            allow_overlaps_to_escape_local_minima=False,
        )
        
        score = float(best_L * best_L / n)
        solution_pool.append({
            'layout': best_layout,
            'side_length': best_L,
            'score': score,
            'generation': 0
        })
        
        print(f'Solution {i+1}: side_length = {best_L:.12f}, score = {score:.12f}')
    
    # Sort by score
    solution_pool.sort(key=lambda x: x['score'])
    print(f'\n=== Solution pool generated ===')
    print(f'Best score in pool: {solution_pool[0]["score"]:.12f}')
    print(f'Worst score in pool: {solution_pool[-1]["score"]:.12f}')
    
    return solution_pool


def crossover_solutions_with_ortools(parent1_layout, parent1_L, parent2_layout, parent2_L, 
                                     target_size, target_L, max_retries=3, 
                                     retry_growth_factor=1.01,
                                     num_copies=10, 
                                     neighborhood_modifiers=[(0.05, 2, 0.7)],
                                     threads=8, time_limit=120, allow_overlaps=False):
    """
    Combine two parent solutions using OR-Tools with a target side length.
    Tries to fit into target_L, with retries at slightly larger sizes if it fails.
    
    Parameters:
    - target_L: the desired side length to fit into
    - max_retries: number of times to retry with larger L if it fails
    - retry_growth_factor: how much to grow L for each retry
    """
    # Combine trees from both parents
    combined_trees = copy.deepcopy(parent1_layout) + copy.deepcopy(parent2_layout)
    
    print(f'  Crossover: {len(parent1_layout)} + {len(parent2_layout)} trees -> select {target_size}, target_L={target_L:.6f}')
    
    # Try with target_L first, then retry with slightly larger sizes
    for retry in range(max_retries + 1):
        current_L = target_L * (retry_growth_factor ** retry)
        
        if retry > 0:
            print(f'    Retry {retry}/{max_retries}: trying with L={current_L:.6f}')
        
        # Use OR-Tools to select best combination
        layout, num_overlaps = solve_with_ortools(
            combined_trees, 
            current_L, 
            target_size, 
            num_copies=num_copies,
            neighborhood_modifiers=neighborhood_modifiers,
            threads=threads,
            time_limit=time_limit,
            allow_overlaps=allow_overlaps
        )
        
        if len(layout) == target_size and num_overlaps == 0:
            # Normalize and get actual side length
            layout, side_length = normalize_trees(layout)
            print(f'    Success! Actual side_length after normalization: {side_length:.6f}')
            return layout, side_length, True
    
    print(f'    Failed after {max_retries} retries')
    return None, None, False


def mutate_solution(layout, max_L, mutation_rate=0.1, mutation_strength=0.05):
    """Apply random mutations to a solution."""
    mutated_layout = copy.deepcopy(layout)
    
    for tree in mutated_layout:
        if random.random() < mutation_rate:
            dx = random.uniform(-mutation_strength, mutation_strength)
            dy = random.uniform(-mutation_strength, mutation_strength)
            dangle = random.uniform(-5, 5)
            tree.move_by(dx, dy, dangle)
    
    # Normalize back
    mutated_layout, side_length = normalize_trees(mutated_layout)
    return mutated_layout, side_length


def tournament_selection(population, tournament_size=3):
    """Select an individual using tournament selection."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return min(tournament, key=lambda x: x['score'])


def find_min_square_ga(n, 
                       initial_pool_size=20,
                       generations=50,
                       population_size=15,
                       num_crossover_copies=10,
                       mutation_rate=0.1,
                       mutation_prob=0.3,
                       elitism_count=3,
                       tournament_size=3,
                       crossover_time_limit=120,
                       crossover_max_retries=3,
                       crossover_retry_growth=1.01,
                       shrink_factor=0.995,
                       threads=8,
                       allow_overlaps_in_crossover=False,
                       save_best_every=5):
    """
    Use a genetic algorithm to evolve solutions by combining them with OR-Tools.
    Each generation forces solutions into progressively smaller squares.
    
    Parameters:
    - n: number of trees
    - initial_pool_size: how many diverse solutions to generate initially
    - generations: number of GA generations to run
    - population_size: size of population to maintain
    - num_crossover_copies: number of perturbed copies when doing crossover
    - mutation_rate: probability of mutating each tree in mutation operation
    - mutation_prob: probability of applying mutation to offspring
    - elitism_count: number of best solutions to keep unchanged
    - tournament_size: size of tournament for selection
    - crossover_time_limit: time limit for OR-Tools crossover operation
    - crossover_max_retries: number of retries for crossover with larger L
    - crossover_retry_growth: growth factor for L on each retry
    - shrink_factor: factor to shrink target L each generation (e.g., 0.995)
    - threads: number of threads for OR-Tools
    - allow_overlaps_in_crossover: whether to allow overlaps to escape local minima
    - save_best_every: save best solution every N generations
    """
    
    print(f'\n{"="*80}')
    print(f'GENETIC ALGORITHM FOR {n} TREES')
    print(f'{"="*80}\n')
    
    # Step 1: Generate initial solution pool
    print('PHASE 1: Generating initial solution pool')
    population = generate_solution_pool(n, pool_size=initial_pool_size, num_copies=10)
    
    # Keep only the best population_size solutions
    population = population[:population_size]
    
    best_ever_solution = copy.deepcopy(population[0])
    print(f'\nInitial best score: {best_ever_solution["score"]:.12f}')
    print(f'Initial best side_length: {best_ever_solution["side_length"]:.12f}')
    
    # Initialize target side length based on best solution
    current_target_L = best_ever_solution['side_length']*shrink_factor
    
    # Step 2: Run genetic algorithm
    print(f'\n{"="*80}')
    print('PHASE 2: Genetic Algorithm Evolution')
    print(f'{"="*80}\n')
    
    neighborhood_modifiers = [(0.03, 1.5, 0),
                              (0.05, 2, 0.7),
                              (0.08, 5, 0.3),
                              (0.12, 6, 0.3)]
    
    for gen in range(generations):
        print(f'\n--- Generation {gen + 1}/{generations} ---')
        
        # Sort population by score
        population.sort(key=lambda x: x['score'])
        
        current_best = population[0]['score']
        current_best_L = population[0]['side_length']
        current_avg = sum(p['score'] for p in population) / len(population)
        
        # Apply shrink factor to target L for this generation
        if gen > 0:  # Don't shrink on first generation
            current_target_L = current_best_L*shrink_factor

        print(f'Target side length for this generation: {current_target_L:.12f}')

        print(f'Population stats: Best={current_best:.12f} (L={current_best_L:.12f}), Avg={current_avg:.12f}, Worst={population[-1]["score"]:.12f}')
        
        # Update best ever
        if population[0]['score'] < best_ever_solution['score']:
            best_ever_solution = copy.deepcopy(population[0])
            print(f'*** NEW BEST SOLUTION FOUND! Score: {best_ever_solution["score"]:.12f}, L={best_ever_solution["side_length"]:.12f} ***')
        
        # Save best solution periodically
        if (gen + 1) % save_best_every == 0:
            output_rows = []
            for i, tree in enumerate(best_ever_solution['layout']):
                output_rows.append({
                    'id': f'{n:03d}_{i}',
                    'x': f's{tree.center_x:.12f}',
                    'y': f's{tree.center_y:.12f}',
                    'deg': f's{tree.angle:.12f}'
                })
            output_df = pd.DataFrame(output_rows)
            output_path = f'solutions/ga_solution_{n}_trees_gen{gen+1}.csv'
            output_df.to_csv(output_path, index=False)
            print(f'Saved checkpoint to {output_path}')
        
        # Create next generation
        next_generation = []
        
        # Elitism: keep best solutions
        for i in range(min(elitism_count, len(population))):
            elite = copy.deepcopy(population[i])
            elite['generation'] = gen + 1
            next_generation.append(elite)
        
        # Generate offspring
        offspring_count = 0
        attempts = 0
        max_attempts = population_size * 5
        
        while len(next_generation) < population_size and attempts < max_attempts:
            attempts += 1
            
            # Select parents
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            # Crossover
            print(f'  Attempt {attempts}: Crossover between solutions with scores {parent1["score"]:.6f} and {parent2["score"]:.6f}')
            
            offspring_layout, offspring_L, success = crossover_solutions_with_ortools(
                parent1['layout'], parent1['side_length'],
                parent2['layout'], parent2['side_length'],
                target_L=current_target_L,
                target_size=n,
                num_copies=num_crossover_copies,
                neighborhood_modifiers=neighborhood_modifiers,
                threads=threads,
                time_limit=crossover_time_limit,
                allow_overlaps=allow_overlaps_in_crossover,
                max_retries=crossover_max_retries,
                retry_growth_factor=crossover_retry_growth
            )
            
            if success:
                # Optional mutation
                if random.random() < mutation_prob:
                    print(f'    Applying mutation...')
                    offspring_layout, offspring_L = mutate_solution(
                        offspring_layout, 
                        offspring_L, 
                        mutation_rate=mutation_rate,
                        mutation_strength=0.05
                    )
                
                offspring_score = float(offspring_L * offspring_L / n)
                
                next_generation.append({
                    'layout': offspring_layout,
                    'side_length': offspring_L,
                    'score': offspring_score,
                    'generation': gen + 1
                })
                
                offspring_count += 1
                print(f'    Success! Offspring score: {offspring_score:.6f} ({len(next_generation)}/{population_size})')
            else:
                print(f'    Failed to generate valid offspring')
        
        print(f'Generated {offspring_count} new offspring in {attempts} attempts')
        
        # If we couldn't generate enough offspring, fill with mutated versions of best solutions
        while len(next_generation) < population_size:
            parent = random.choice(population[:elitism_count + 2])
            mutated_layout, mutated_L = mutate_solution(
                parent['layout'],
                parent['side_length'],
                mutation_rate=0.2,
                mutation_strength=0.1
            )
            mutated_score = float(mutated_L * mutated_L / n)
            next_generation.append({
                'layout': mutated_layout,
                'side_length': mutated_L,
                'score': mutated_score,
                'generation': gen + 1
            })
            print(f'  Filled with mutation: score {mutated_score:.6f}')
        
        population = next_generation
    
    # Final report
    print(f'\n{"="*80}')
    print('GENETIC ALGORITHM COMPLETE')
    print(f'{"="*80}')
    print(f'Best solution found: score = {best_ever_solution["score"]:.12f}')
    print(f'Side length: {best_ever_solution["side_length"]:.12f}')
    print(f'Generation: {best_ever_solution["generation"]}')
    
    return best_ever_solution['side_length'], best_ever_solution['layout']


i = 30
print('------------------ Working with',i,'christmas trees ------------------')
num_copies = 700//i
# improve_solution(i, num_copies=20)

# Run genetic algorithm
best_L, best_layout = find_min_square_ga(
    n=i,
    initial_pool_size=10,      # Generate 10 diverse initial solutions
    generations=200,            # Run 200 generations
    population_size=8,          # Maintain population of 8
    num_crossover_copies=15,    # Use 15 copies in crossover
    mutation_rate=0.15,         # 15% chance to mutate each tree
    mutation_prob=0.3,          # 30% chance to mutate offspring
    elitism_count=4,            # Keep 5 best solutions
    tournament_size=3,          # Tournament size of 3
    crossover_time_limit=180,   # 3 minutes for crossover
    threads=8,
    allow_overlaps_in_crossover=False,
    save_best_every=5,          # Save every 5 generations
    crossover_max_retries=3,    # Try up to 3 times with larger L if crossover fails
    crossover_retry_growth=1.01,# Increase L by 1% for each retry
    shrink_factor=0.999         # Shrink target L by 0.1% each generation
)

# Save final best solution
output_rows = []
for idx, tree in enumerate(best_layout):
    output_rows.append({
        'id': f'{i:03d}_{idx}',
        'x': f's{tree.center_x:.12f}',
        'y': f's{tree.center_y:.12f}',
        'deg': f's{tree.angle:.12f}'
    })
output_df = pd.DataFrame(output_rows)
output_path = f'solutions/ga_solution_{i}_trees_final.csv'
output_df.to_csv(output_path, index=False)
print(f'\nFinal solution saved to {output_path}')
print(f'Final score: {best_L*best_L/i:.12f}')
