use std::error::Error;
use std::f64::consts::PI;
use std::fs::File;
use std::path::Path;

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

fn total_overlap_area(trees: &[ChristmasTree]) -> f64 {
    let mut area = 0.0_f64;
    let n = trees.len();
    for i in 0..n {
        for j in (i + 1)..n {
            // Use geo's BooleanOps for intersection
            let inter = trees[i].polygon.intersection(&trees[j].polygon);
            let a = inter.unsigned_area();
            if a > 0.0 {
                area += a;
            }
        }
    }
    area
}

fn boundary_violation(trees: &[ChristmasTree], l: f64) -> f64 {
    let mut penalty = 0.0_f64;
    for t in trees {
        if let Some(rect) = t.polygon.bounding_rect() {
            let minx = rect.min().x;
            let miny = rect.min().y;
            let maxx = rect.max().x;
            let maxy = rect.max().y;

            if minx < 0.0 {
                penalty += (0.0 - minx).powi(2);
            }
            if miny < 0.0 {
                penalty += (0.0 - miny).powi(2);
            }
            if maxx > l {
                penalty += (maxx - l).powi(2);
            }
            if maxy > l {
                penalty += (maxy - l).powi(2);
            }
        }
    }
    penalty
}

fn energy(trees: &[ChristmasTree], l: f64) -> f64 {
    BIG * total_overlap_area(trees) + BIG * boundary_violation(trees, l) * 2.0
}

#[derive(Clone, Copy)]
struct AnnealParams {
    n_steps: usize,
    t_start: f64,
    t_end: f64,
    step_xy: f64,
    step_angle: f64,
    shrink_bias: f64,
}

fn biased_delta(step_size: f64, shrink_bias: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen_range(-step_size..=step_size);
    // if random variable is smaller than shrink_bias, then chose r between 0 and step_size, otherwise, chose r between -step_size and step_size
    let bias: f64 = if rng.gen_range(0.0..=1.0) < shrink_bias {
        -1.0
    } else {
        0.0
    };
    r + bias * step_size
}

fn anneal_layout(
    trees: &[ChristmasTree],
    l: f64,
    params: AnnealParams,
) -> (Vec<ChristmasTree>, f64) {
    let mut rng = rand::thread_rng();
    let mut current: Vec<ChristmasTree> = trees.to_vec();
    let mut best: Vec<ChristmasTree> = current.clone();

    let mut e_current = energy(&current, l);
    let mut e_best = e_current;

    for k in 0..params.n_steps {
        // Early exit if we've reached perfect energy
        if e_current <= 0.0 {
            println!(
                "Perfect solution found at step {}/{}! Energy: {:.6}",
                k, params.n_steps, e_current
            );
            break;
        }

        if k % 500 == 0 {
            println!(
                "Step {}/{}, current energy: {:.6}, best energy: {:.6}",
                k, params.n_steps, e_current, e_best
            );
        }

        // Exponential temperature schedule
        let denom = (params.n_steps.saturating_sub(1)).max(1) as f64;
        let t = k as f64 / denom;
        let temp = params.t_start * (params.t_end / params.t_start).powf(t);

        // Pick random tree and propose move
        let idx = rng.gen_range(0..current.len());
        let old_tree = current[idx].clone();

        let dx = biased_delta(params.step_xy, params.shrink_bias);
        let dy = biased_delta(params.step_xy, params.shrink_bias);
        let dangle = rng.gen_range(-params.step_angle..=params.step_angle);

        current[idx].move_by(dx, dy, dangle);

        let e_new = energy(&current, l);
        let d_e = e_new - e_current;

        let accept = if d_e <= 0.0 {
            true
        } else {
            // raw identifier because `gen` is a reserved keyword
            let r: f64 = rng.r#gen();
            r < (-d_e / temp).exp()
        };

        if accept {
            e_current = e_new;
            if e_new <= e_best {
                e_best = e_new;
                best = current.clone();
            }
        } else {
            // reject move
            current[idx] = old_tree;
        }
    }

    (best, e_best)
}

fn find_min_square(
    trees: &[ChristmasTree],
    mut l: f64,
    shrink_factors: &[f64],
    max_retries: usize,
    min_l: f64,
    params: AnnealParams,
) -> (f64, Vec<ChristmasTree>) {
    let mut current_layout = trees.to_vec();
    let mut best_l = l;
    let mut best_layout = trees.to_vec();
    
    // Start with the first (most aggressive) shrink factor
    let mut shrink_factor_idx = 0;
    let mut current_shrink_factor = shrink_factors[shrink_factor_idx];
    
    println!("Starting with shrink factor: {:.6}", current_shrink_factor);

    while l > min_l {
        println!("Trying with side length = {:.12}, shrink factor = {:.6}", l, current_shrink_factor);
        let (layout, e) = anneal_layout(&current_layout, l, params);
        if e < 1e-15 {
            // success
            best_l = l;
            best_layout = layout.clone();
            l *= current_shrink_factor;
            current_layout = layout;
        } else {
            println!(
                "Failed with L = {:.12} (E = {:.6}). Attempting retries...",
                l, e
            );
            let mut success = false;
            for retry in 1..=max_retries {
                let l_retry =
                    l * (1.0 + (1.0 - current_shrink_factor) / (max_retries as f64 + 1.0) * retry as f64);
                println!(
                    "Retry {}/{} with slightly larger L = {:.12}",
                    retry, max_retries, l_retry
                );
                let (layout_retry, e_retry) =
                    anneal_layout(&current_layout, l_retry, params);
                if e_retry < 1e-15 {
                    println!(
                        "Retry {} succeeded! Continuing from L = {:.12}",
                        retry, l_retry
                    );
                    best_l = l_retry;
                    best_layout = layout_retry.clone();
                    l = l_retry * current_shrink_factor;
                    current_layout = layout_retry;
                    success = true;
                    break;
                } else {
                    println!(
                        "Retry {} failed (E = {:.6})",
                        retry, e_retry
                    );
                }
            }
            if !success {
                // All retries exhausted - try next shrink factor if available
                if shrink_factor_idx + 1 < shrink_factors.len() {
                    shrink_factor_idx += 1;
                    current_shrink_factor = shrink_factors[shrink_factor_idx];
                    println!(
                        "All retries exhausted. Switching to less aggressive shrink factor: {:.6}",
                        current_shrink_factor
                    );
                    // Continue with the new shrink factor instead of breaking
                } else {
                    println!("All retries exhausted and no more shrink factors available. Stopping optimization.");
                    break;
                }
            }
        }
    }

    (best_l, best_layout)
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

fn improve_solution(n: u32) -> Result<(), Box<dyn Error>> {
    let solutions_folder = "solutions3";
    
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

    let params = AnnealParams {
        n_steps: 30_000,
        t_start: 0.8,
        t_end: 1e-15,
        step_xy: 0.04,
        step_angle: 3.0,
        shrink_bias: 0.1,
    };

    let shrink_factors = vec![0.98, 0.99, 0.995, 0.998, 0.999]; // we start with an aggressive shrink factor. When an optimization loop has failed max_retries time, we progress with a less aggressive shrink factor
    let max_retries = 5;

    let (best_l, best_layout) = find_min_square(
        &trees,
        original_side_length,
        &shrink_factors,
        max_retries,
        0.5,
        params,
    );

    let new_score = best_l * best_l / (n as f64);
    
    println!("Original side_length: {:.12}, score: {:.12}", original_side_length, original_score);
    println!("New side_length: {:.12}, score: {:.12}", best_l, new_score);

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

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    for i in 3_u32..70_u32 {
        improve_solution(i)?;
    }
    Ok(())
}
