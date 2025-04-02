# VRP Optimizer Workaround

This README provides instructions for using the workaround scripts to run the VRP Optimizer when you encounter issues with certain command options.

## The Problem

The current version of the VRP Optimizer has some indentation errors and compatibility issues when running with certain combinations of options, particularly:
- `--traffic`
- `--driver_hours`

These options may cause the program to fail with `IndentationError` or other exceptions.

## Workaround Scripts

### 1. Simple Runner (`run_simple.py`)

This script runs the VRP Optimizer with basic options only, avoiding the problematic options:

```bash
# Make executable
chmod +x run_simple.py

# Run with default settings (15 iterations)
./run_simple.py
```

The results will be saved to the `results_simple` directory.

### 2. Robust Runner (`run_vrp.py`)

This script provides more flexibility and error handling:

```bash
# Make executable
chmod +x run_vrp.py

# Run with basic options
./run_vrp.py --iterations 20 --output my_results

# If you try to use the problematic options, it will automatically
# retry without them if the initial command fails
./run_vrp.py --iterations 20 --traffic --driver_hours --output my_results
```

The runner will:
1. Try to run with all specified options
2. If it fails and you've used `--traffic` or `--driver_hours`, it will retry without these options
3. Log detailed information to `vrp_runner.log`

## Viewing Results

After running either script, check the specified output directory for:
- `routes.html`: Map visualization of the routes
- `kpi_dashboard.html`: Performance metrics dashboard
- `progress.png`: Graph showing algorithm performance over iterations
- `iterations.html`: Detailed report of each iteration
- Other visualization files based on enabled options

## Tips

1. Start with fewer iterations (10-20) to test if the configuration works
2. If you need traffic or driver hours constraints, you may need to fix the indentation errors in the source code
3. Check `vrp_runner.log` for detailed error information when using `run_vrp.py`

## Long-Term Fix

To properly fix the issues:
1. Check indentation in `src/utils/solution.py` (particularly around lines 703, 739, 847, 854, and 1133)
2. Ensure consistent use of tabs vs. spaces throughout the codebase
3. Add proper error handling around traffic and driver hours constraint implementations 