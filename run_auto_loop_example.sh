#!/bin/bash

# Example script to run the solver in auto-loop mode
# This will try different row/col configurations automatically

python3.13 solver_sa_patterns.py \
  --t1x 0 \
  --t1y 0 \
  --t1a 250 \
  --t2x 0.8 \
  --t2y 0.3 \
  --t2a 90 \
  --row_x_offset k0 \
  --row_y_offset 0.8 \
  --col_x_offset 1.2 \
  --col_y_offset k0 \
  --auto_loop \
  --config_file autoloop.config \
  --optimize \
  --sa_iters 10000 \
  --sa_log_interval 100

echo "Done! Check the solutions/ directory for results."

python3.13 solver_sa_patterns.py \
  --t1x 0 \
  --t1y 0 \
  --t1a 250 \
  --t2x 0.8 \
  --t2y 0.3 \
  --t2a 90 \
  --row_x_offset k0 \
  --row_y_offset 0.8 \
  --col_x_offset 1.2 \
  --col_y_offset k0 \
  --auto_loop \
  --config_file autoloop.config \
  --optimize \
  --sa_iters 30000 \
  --sa_log_interval 100

echo "Done! Check the solutions/ directory for results."

python3.13 solver_sa_patterns.py \
  --t1x 0 \
  --t1y 0 \
  --t1a 250 \
  --t2x 0.8 \
  --t2y 0.3 \
  --t2a 90 \
  --row_x_offset k0 \
  --row_y_offset 0.8 \
  --col_x_offset 1.2 \
  --col_y_offset k0 \
  --auto_loop \
  --config_file autoloop.config \
  --optimize \
  --sa_iters 30000 \
  --sa_log_interval 100

echo "Done! Check the solutions/ directory for results."

python3.13 solver_sa_patterns.py \
  --t1x 0 \
  --t1y 0 \
  --t1a 250 \
  --t2x 0.8 \
  --t2y 0.3 \
  --t2a 90 \
  --row_x_offset k0 \
  --row_y_offset 0.8 \
  --col_x_offset 1.2 \
  --col_y_offset k0 \
  --auto_loop \
  --config_file autoloop.config \
  --optimize \
  --sa_iters 30000 \
  --sa_log_interval 100

echo "Done! Check the solutions/ directory for results."

python3.13 solver_sa_patterns.py \
  --t1x 0 \
  --t1y 0 \
  --t1a 250 \
  --t2x 0.8 \
  --t2y 0.3 \
  --t2a 90 \
  --row_x_offset k0 \
  --row_y_offset 0.8 \
  --col_x_offset 1.2 \
  --col_y_offset k0 \
  --auto_loop \
  --config_file autoloop.config \
  --optimize \
  --sa_iters 30000 \
  --sa_log_interval 100

echo "Done! Check the solutions/ directory for results."