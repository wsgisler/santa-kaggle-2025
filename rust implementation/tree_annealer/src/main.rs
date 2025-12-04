use std::error::Error;
use std::f64::consts::PI;
use std::fs::File;
use std::path::Path;
use std::time::Instant;
use std::io::Write;

use csv::{ReaderBuilder, WriterBuilder};
use geo::{Coord, Polygon, LineString};
use geo::algorithm::area::Area;
use geo::algorithm::bounding_rect::BoundingRect;
use geo::algorithm::bool_ops::BooleanOps;
use rand::Rng;
use serde::{Deserialize, Serialize};

const BIG: f64 = 1e6;

#[derive(Clone, Debug)]
struct ChristmasTree {
    center_x: f64,
    center_y: f64,
    angle_deg: f64,
    polygon: Polygon<f64>,
}

impl ChristmasTree {
    fn new(center_x: f64, center_y: f64, angle_deg: f64) -> Self {
        let mut t = ChristmasTree {
            center_x,
            center_y,
            angle_deg,
            // create an empty polygon with explicit type
            polygon: Polygon::new(LineString::<f64>::new(vec![]), vec![]),
        };
        t.build_polygon();
        t
    }

    fn build_polygon(&mut self) {
        // Values correspond to the Python Decimal values (without scaling)
        let trunk_w = 0.15_f64;
        let trunk_h = 0.2_f64;
        let base_w = 0.7_f64;
        let mid_w = 0.4_f64;
        let top_w = 0.25_f64;
        let tip_y = 0.8_f64;
        let tier_1_y = 0.5_f64;
        let tier_2_y = 0.25_f64;
        let base_y = 0.0_f64;
        let trunk_bottom_y = -trunk_h;

        // Polygon around origin, matching the Python point list
        let mut coords = vec![
            (0.0, tip_y),
            (top_w / 2.0, tier_1_y),
            (top_w / 4.0, tier_1_y),
            (mid_w / 2.0, tier_2_y),
            (mid_w / 4.0, tier_2_y),
            (base_w / 2.0, base_y),
            (trunk_w / 2.0, base_y),
            (trunk_w / 2.0, trunk_bottom_y),
            (-trunk_w / 2.0, trunk_bottom_y),
            (-trunk_w / 2.0, base_y),
            (-base_w / 2.0, base_y),
            (-mid_w / 4.0, tier_2_y),
            (-mid_w / 2.0, tier_2_y),
            (-top_w / 4.0, tier_1_y),
            (-top_w / 2.0, tier_1_y),
        ];

        // Rotate around (0, 0) by angle_deg
        let angle_rad = self.angle_deg * PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        for (x, y) in &mut coords {
            let rx = cos_a * *x - sin_a * *y;
            let ry = sin_a * *x + cos_a * *y;
            *x = rx;
            *y = ry;
        }

        // Translate by center_x, center_y
        let exterior: Vec<Coord<f64>> = coords
            .into_iter()
            .map(|(x, y)| Coord {
                x: x + self.center_x,
                y: y + self.center_y,
            })
            .collect();

        self.polygon = Polygon::new(exterior.into(), vec![]);
    }

    fn move_by(&mut self, dx: f64, dy: f64, dangle: f64) {
        self.center_x += dx;
        self.center_y += dy;
        self.angle_deg += dangle;
        self.build_polygon();
    }
}

fn overlap_stats(trees: &[ChristmasTree]) -> (f64, usize) {
    let mut area = 0.0_f64;
    let mut count = 0_usize;
    let n = trees.len();
    for i in 0..n {
        for j in (i + 1)..n {
            // Use geo's BooleanOps for intersection
            let inter = trees[i].polygon.intersection(&trees[j].polygon);
            let a = inter.unsigned_area();
            if a > 0.0 {
                area += a;
                count += 1;
            }
        }
    }
    (area, count)
}

fn compute_bounding_side_length(trees: &[ChristmasTree]) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees {
        if let Some(rect) = t.polygon.bounding_rect() {
            min_x = min_x.min(rect.min().x);
            min_y = min_y.min(rect.min().y);
            max_x = max_x.max(rect.max().x);
            max_y = max_y.max(rect.max().y);
        }
    }

    let width = max_x - min_x;
    let height = max_y - min_y;
    width.max(height)
}

fn energy(trees: &[ChristmasTree]) -> f64 {
    let (overlap_area, _) = overlap_stats(trees);
    let bounding_side = compute_bounding_side_length(trees);
    
    // Primary objective: minimize bounding box side length
    // Secondary objective: penalize overlaps with a small weight
    bounding_side + 100.0 * overlap_area
}

/// Note: For the new energy function, we can't easily compute incremental updates
/// because bounding box depends on all trees. We'll just recompute full energy.
/// This is a placeholder that won't be used in the new approach.
fn _energy_single_tree_deprecated(tree: &ChristmasTree, other_trees: &[ChristmasTree], tree_idx: usize) -> f64 {
    let mut overlap_area = 0.0_f64;
    
    // Check overlaps with all other trees
    for (j, other) in other_trees.iter().enumerate() {
        if j != tree_idx {
            let inter = tree.polygon.intersection(&other.polygon);
            let a = inter.unsigned_area();
            if a > 0.0 {
                overlap_area += a;
            }
        }
    }
    
    10.0 * overlap_area
}

#[derive(Clone, Copy, Debug)]
struct AnnealParams {
    n_steps: usize,
    t_start: f64,
    t_end: f64,
    step_xy: f64,
    step_angle: f64,
}

fn random_delta(step_size: f64) -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen_range(-step_size..=step_size)
}

fn anneal_layout(
    trees: &[ChristmasTree],
    params: AnnealParams,
) -> (Vec<ChristmasTree>, f64, Option<(Vec<ChristmasTree>, f64, f64)>) {
    let mut rng = rand::thread_rng();
    let mut current: Vec<ChristmasTree> = trees.to_vec();
    let mut best: Vec<ChristmasTree> = current.clone();
    
    let n_trees = trees.len() as f64;

    let mut e_current = energy(&current);
    let mut e_best = e_current;
    
    // Track best non-overlapping solution
    let mut best_non_overlap: Option<(Vec<ChristmasTree>, f64, f64)> = None;
    
    // Check if initial solution has no overlaps
    let (initial_overlap, _) = overlap_stats(&current);
    if initial_overlap == 0.0 {
        let side = compute_bounding_side_length(&current);
        let score = side * side / n_trees;
        best_non_overlap = Some((current.clone(), side, score));
    }

    for k in 0..params.n_steps {
        if k % 500 == 0 {
            let (curr_overlap, overlap_pairs) = overlap_stats(&current);
            let curr_side = compute_bounding_side_length(&current);
            let curr_score = curr_side * curr_side / n_trees;
            
            print!(
                "Step {}/{}, energy: {:.6}, side: {:.6}, overlaps: {} pairs, ",
                k, params.n_steps, e_current, curr_side, overlap_pairs
            );
            
            if let Some((_, best_no_side, best_no_score)) = &best_non_overlap {
                println!("best_non_overlap: side={:.12}, score={:.12}", best_no_side, best_no_score);
            } else {
                println!("best_non_overlap: None");
            }
        }

        // Exponential temperature schedule
        let denom = (params.n_steps.saturating_sub(1)).max(1) as f64;
        let t = k as f64 / denom;
        let temp = params.t_start * (params.t_end / params.t_start).powf(t);

        // Pick random tree and propose move
        let idx = rng.gen_range(0..current.len());
        let old_tree = current[idx].clone();

        let dx = random_delta(params.step_xy);
        let dy = random_delta(params.step_xy);
        let dangle = random_delta(params.step_angle);

        current[idx].move_by(dx, dy, dangle);

        // Recompute full energy (can't do incremental with bounding box)
        let e_new = energy(&current);
        let d_e = e_new - e_current;

        let accept = if d_e <= 0.0 {
            true
        } else {
            let r: f64 = rng.r#gen();
            r < (-d_e / temp).exp()
        };

        if accept {
            e_current = e_new;
            if e_new <= e_best {
                e_best = e_new;
                best = current.clone();
            }
            
            // Check if current solution has no overlaps and update best_non_overlap
            let (overlap_area, _) = overlap_stats(&current);
            if overlap_area == 0.0 {
                let side = compute_bounding_side_length(&current);
                let score = side * side / n_trees;
                
                let should_update = match &best_non_overlap {
                    None => true,
                    Some((_, _, best_score)) => score < *best_score,
                };
                
                if should_update {
                    best_non_overlap = Some((current.clone(), side, score));
                }
            }
        } else {
            // reject move
            current[idx] = old_tree;
        }
    }

    (best, e_best, best_non_overlap)
}

fn optimize_packing(
    trees: &[ChristmasTree],
    params: AnnealParams,
) -> (Vec<ChristmasTree>, f64, f64) {
    println!("\nStarting optimization...");
    println!("Parameters:");
    println!("  n_steps: {}", params.n_steps);
    println!("  t_start: {}", params.t_start);
    println!("  t_end: {}", params.t_end);
    println!("  step_xy: {}", params.step_xy);
    println!("  step_angle: {}", params.step_angle);
    
    let (best_layout, _best_energy, best_non_overlap) = anneal_layout(trees, params);
    
    // Return the best non-overlapping solution if found, otherwise return best overall
    match best_non_overlap {
        Some((layout, side, score)) => {
            println!("\nFound non-overlapping solution!");
            println!("  Side length: {:.12}", side);
            println!("  Score: {:.12}", score);
            (layout, side, score)
        }
        None => {
            println!("\nNo non-overlapping solution found. Returning best overall layout.");
            let side = compute_bounding_side_length(&best_layout);
            let score = side * side / (trees.len() as f64);
            let (overlap_area, overlap_pairs) = overlap_stats(&best_layout);
            println!("  Side length: {:.12}", side);
            println!("  Score: {:.12}", score);
            println!("  Overlap area: {:.6}", overlap_area);
            println!("  Overlap pairs: {}", overlap_pairs);
            (best_layout, side, score)
        }
    }
}

fn normalize_trees(trees: &mut [ChristmasTree]) -> f64 {
    // Translate so all fit into first quadrant, return side length
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;

    for t in trees.iter() {
        if let Some(rect) = t.polygon.bounding_rect() {
            if rect.min().x < min_x {
                min_x = rect.min().x;
            }
            if rect.min().y < min_y {
                min_y = rect.min().y;
            }
        }
    }

    for t in trees.iter_mut() {
        t.move_by(-min_x, -min_y, 0.0);
    }

    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees.iter() {
        if let Some(rect) = t.polygon.bounding_rect() {
            if rect.max().x > max_x {
                max_x = rect.max().x;
            }
            if rect.max().y > max_y {
                max_y = rect.max().y;
            }
        }
    }

    max_x.max(max_y)
}

#[derive(Debug, Deserialize)]
struct InputRow {
    id: String,
    x: String,
    y: String,
    deg: String,
}

#[derive(Debug, Serialize)]
struct OutputRow {
    id: String,
    x: String,
    y: String,
    deg: String,
}

#[derive(Debug, Clone)]
struct TuningResult {
    params: AnnealParams,
    final_score: f64,
    final_side_length: f64,
    duration_secs: f64,
    improved: bool,
}

fn improve_solution(n: u32) -> Result<(), Box<dyn Error>> {
    improve_solution_with_params(n, None, None, None)?;
    Ok(())
}

fn improve_solution_with_params(
    n: u32,
    params_override: Option<AnnealParams>,
    _shrink_factors_override: Option<Vec<f64>>,
    _max_retries_override: Option<usize>,
) -> Result<TuningResult, Box<dyn Error>> {
    let solutions_folder = "solutions1";
    
    let solution_path = format!("{}/solution_{}_trees.csv", solutions_folder, n);
    let path = Path::new(&solution_path);

    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut trees: Vec<ChristmasTree> = Vec::new();

    for result in rdr.deserialize::<InputRow>() {
        let row = result?;

        // Strip leading 's' and parse
        let center_x: f64 = row.x.trim_start_matches('s').parse()?;
        let center_y: f64 = row.y.trim_start_matches('s').parse()?;
        let angle_deg: f64 = row.deg.trim_start_matches('s').parse()?;

        let tree = ChristmasTree::new(center_x, center_y, angle_deg);
        trees.push(tree);
    }

    // Normalize positions
    let original_side_length = normalize_trees(&mut trees);
    let original_score = original_side_length * original_side_length / (n as f64);

    // Use provided parameters or defaults
    let params = params_override.unwrap_or(AnnealParams {
        n_steps: 100_000,
        t_start: 1.0,
        t_end: 1e-15,
        step_xy: 0.05,
        step_angle: 5.0,
    });

    let start_time = Instant::now();

    let (best_layout, best_l, new_score) = optimize_packing(&trees, params);

    let duration = start_time.elapsed();
    
    println!("Original side_length: {:.12}, score: {:.12}", original_side_length, original_score);
    println!("New side_length: {:.12}, score: {:.12}", best_l, new_score);
    println!("Time taken: {:.2}s", duration.as_secs_f64());

    // Check if sa_solution file already exists
    let output_path = format!("{}/sa_solution_{}_trees.csv", solutions_folder, n);
    let existing_sa_score = if Path::new(&output_path).exists() {
        println!("Existing sa_solution file found. Checking its score...");
        
        // Read existing sa_solution file
        let sa_file = File::open(&output_path)?;
        let mut sa_rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(sa_file);
        
        let mut sa_trees: Vec<ChristmasTree> = Vec::new();
        for result in sa_rdr.deserialize::<InputRow>() {
            let row = result?;
            let center_x: f64 = row.x.trim_start_matches('s').parse()?;
            let center_y: f64 = row.y.trim_start_matches('s').parse()?;
            let angle_deg: f64 = row.deg.trim_start_matches('s').parse()?;
            let tree = ChristmasTree::new(center_x, center_y, angle_deg);
            sa_trees.push(tree);
        }
        
        let sa_side_length = normalize_trees(&mut sa_trees);
        let sa_score = sa_side_length * sa_side_length / (n as f64);
        println!("Existing sa_solution score: {:.12}", sa_score);
        Some(sa_score)
    } else {
        None
    };

    // Determine if we should write the file
    let should_write = match existing_sa_score {
        Some(sa_score) => {
            if new_score < sa_score {
                let improvement = ((sa_score - new_score) / sa_score) * 100.0;
                println!("New solution is {:.2}% better than existing sa_solution!", improvement);
                true
            } else {
                println!("Existing sa_solution is better or equal. Not overwriting.");
                false
            }
        },
        None => {
            // No existing sa_solution, check against original
            if new_score < original_score {
                let improvement = ((original_score - new_score) / original_score) * 100.0;
                println!("Improvement found! {:.2}% better than original.", improvement);
                true
            } else {
                println!("No improvement over original solution. Not writing file.");
                false
            }
        }
    };

    if should_write {
        println!("Writing to file...");
        let out_file = File::create(&output_path)?;
        let mut wtr = WriterBuilder::new().from_writer(out_file);

        for (i, tree) in best_layout.iter().enumerate() {
            let row = OutputRow {
                id: format!("{:03}_{}", n, i),
                x: format!("s{:.12}", tree.center_x),
                y: format!("s{:.12}", tree.center_y),
                deg: format!("s{:.12}", tree.angle_deg),
            };
            wtr.serialize(row)?;
        }

        wtr.flush()?;
        println!("Wrote improved solution to {}.", output_path);
    }

    Ok(TuningResult {
        params,
        final_score: new_score,
        final_side_length: best_l,
        duration_secs: duration.as_secs_f64(),
        improved: should_write,
    })
}

fn tune_parameters(n: u32, num_trials: usize) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "=".repeat(80));
    println!("PARAMETER TUNING FOR {} TREES", n);
    println!("Running {} different parameter combinations", num_trials);
    println!("{}\n", "=".repeat(80));

    let mut all_results: Vec<TuningResult> = Vec::new();
    let mut best_result: Option<TuningResult> = None;
    
    // Define parameter ranges to explore
    let n_steps_options = vec![50_000, 100_000, 150_000, 200_000];
    let t_start_options = vec![1.0, 2.0, 5.0, 10.0];
    let step_xy_options = vec![0.03, 0.05, 0.08, 0.1];
    let step_angle_options = vec![3.0, 5.0, 8.0, 10.0];
    
    // Generate random combinations
    let mut rng = rand::thread_rng();
    
    for trial in 0..num_trials {
        println!("\n{}", "=".repeat(80));
        println!("TRIAL {}/{}", trial + 1, num_trials);
        println!("{}", "=".repeat(80));
        
        // Randomly select parameters
        let n_steps = n_steps_options[rng.gen_range(0..n_steps_options.len())];
        let t_start = t_start_options[rng.gen_range(0..t_start_options.len())];
        let step_xy = step_xy_options[rng.gen_range(0..step_xy_options.len())];
        let step_angle = step_angle_options[rng.gen_range(0..step_angle_options.len())];
        
        let params = AnnealParams {
            n_steps,
            t_start,
            t_end: 1e-15,
            step_xy,
            step_angle,
        };
        
        println!("Testing parameters:");
        println!("  n_steps: {}", n_steps);
        println!("  t_start: {}", t_start);
        println!("  step_xy: {}", step_xy);
        println!("  step_angle: {}", step_angle);
        
        match improve_solution_with_params(n, Some(params), None, None) {
            Ok(result) => {
                println!("\nResult:");
                println!("  Final score: {:.12}", result.final_score);
                println!("  Final side_length: {:.12}", result.final_side_length);
                println!("  Duration: {:.2}s", result.duration_secs);
                println!("  Improved: {}", result.improved);
                
                // Update best result
                let is_best = match &best_result {
                    None => true,
                    Some(best) => result.final_score < best.final_score,
                };
                
                if is_best {
                    println!("\n*** NEW BEST RESULT! ***");
                    best_result = Some(result.clone());
                }
                
                all_results.push(result);
            }
            Err(e) => {
                println!("Trial failed with error: {}", e);
            }
        }
    }
    
    // Write results summary
    println!("\n\n{}", "=".repeat(80));
    println!("TUNING COMPLETE");
    println!("{}\n", "=".repeat(80));
    
    if let Some(best) = &best_result {
        println!("BEST PARAMETERS FOUND:");
        println!("  n_steps: {}", best.params.n_steps);
        println!("  t_start: {}", best.params.t_start);
        println!("  t_end: {}", best.params.t_end);
        println!("  step_xy: {}", best.params.step_xy);
        println!("  step_angle: {}", best.params.step_angle);
        println!("\nBEST RESULT:");
        println!("  Final score: {:.12}", best.final_score);
        println!("  Final side_length: {:.12}", best.final_side_length);
        println!("  Duration: {:.2}s", best.duration_secs);
        
        // Write detailed results to file
        let results_path = format!("tuning_results_{}_trees.txt", n);
        let mut file = File::create(&results_path)?;
        
        writeln!(file, "Parameter Tuning Results for {} Trees", n)?;
        writeln!(file, "Total trials: {}", num_trials)?;
        writeln!(file, "{}\n", "=".repeat(80))?;
        
        writeln!(file, "BEST PARAMETERS:")?;
        writeln!(file, "  n_steps: {}", best.params.n_steps)?;
        writeln!(file, "  t_start: {}", best.params.t_start)?;
        writeln!(file, "  t_end: {}", best.params.t_end)?;
        writeln!(file, "  step_xy: {}", best.params.step_xy)?;
        writeln!(file, "  step_angle: {}", best.params.step_angle)?;
        writeln!(file, "\nBEST RESULT:")?;
        writeln!(file, "  Final score: {:.12}", best.final_score)?;
        writeln!(file, "  Final side_length: {:.12}", best.final_side_length)?;
        writeln!(file, "  Duration: {:.2}s\n", best.duration_secs)?;
        
        writeln!(file, "\n{}", "=".repeat(80))?;
        writeln!(file, "ALL RESULTS (sorted by score):\n")?;
        
        // Sort all results by score
        let mut sorted_results = all_results.clone();
        sorted_results.sort_by(|a, b| a.final_score.partial_cmp(&b.final_score).unwrap());
        
        for (i, result) in sorted_results.iter().enumerate() {
            writeln!(file, "Rank {}: score={:.12}, time={:.2}s, n_steps={}, t_start={}, step_xy={}, step_angle={}",
                i + 1,
                result.final_score,
                result.duration_secs,
                result.params.n_steps,
                result.params.t_start,
                result.params.step_xy,
                result.params.step_angle
            )?;
        }
        
        println!("\nDetailed results written to {}", results_path);
    } else {
        println!("No successful trials completed.");
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Run parameter tuning for 30 trees with 15 different parameter combinations
    // tune_parameters(30, 15)?;
    
    for j in 1_u32..=100_u32 {
        improve_solution(28)?;
    }
    // Or to run the regular optimization loop:
    // for i in 26_u32..27_u32 {
    //     improve_solution(i)?;
    // }
    
    Ok(())
}