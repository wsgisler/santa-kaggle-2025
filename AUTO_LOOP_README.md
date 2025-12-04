# Auto-Loop Feature

The solver has been extended with an auto-loop mode that automatically tries different starting configurations (different rows/cols combinations).

## How It Works

- **Number of trees**: For each configuration with `rows` and `cols`, the total number of trees is calculated as `n = 2 × rows × cols`
- **Output files**: Solutions are saved to `solutions/solution_n_trees.csv`
- **Smart saving**: A new solution is only saved if:
  - No file with the same name exists yet, OR
  - The new solution has a smaller side length than the existing solution

## Usage

### Auto-Loop Mode

```bash
python solver_sa_patterns.py \
  --t1x 0 --t1y 0 --t1a 0 \
  --t2x 1.5 --t2y 0 --t2a 0 \
  --row_x_offset 0 --row_y_offset 2.0 \
  --col_x_offset 2.0 --col_y_offset 0 \
  --auto_loop \
  --max_configs 20 \
  --optimize \
  --sa_iters 2000
```

### New Arguments

- `--auto_loop`: Enable automatic configuration loop
- `--max_configs N`: Maximum number of configurations to try (default: 20)
- `--config_file FILE`: Optional file with custom rows,cols pairs (one per line)

### Custom Configuration File

You can provide a custom list of configurations to try by creating a file like:

```
# rows,cols
1,5
2,3
3,4
5,5
10,10
```

Then run:

```bash
python solver_sa_patterns.py ... --auto_loop --config_file my_configs.txt
```

### Original Single-Configuration Mode

The original behavior still works - just specify `--rows` and `--cols`:

```bash
python solver_sa_patterns.py \
  --t1x 0 --t1y 0 --t1a 0 \
  --t2x 1.5 --t2y 0 --t2a 0 \
  --row_x_offset 0 --row_y_offset 2.0 \
  --col_x_offset 2.0 --col_y_offset 0 \
  --rows 5 --cols 5 \
  --output my_solution.csv \
  --optimize
```

## Example

Run the included example script:

```bash
./run_auto_loop_example.sh
```

This will try 20 different configurations and save the best solutions to the `solutions/` directory.
