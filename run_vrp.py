#!/usr/bin/env python3
"""
VRP Optimizer Runner - A robust wrapper for the VRP Optimizer that handles errors gracefully
"""

import os
import sys
import subprocess
import argparse
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vrp_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='VRP Optimizer Runner')
    parser.add_argument('--iterations', type=int, default=500,
                    help='Number of iterations for the ALNS algorithm')
    parser.add_argument('--output', type=str, default='results',
                    help='Directory to save results')
    parser.add_argument('--config', type=str, default='config.json',
                    help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/sample_data.csv',
                    help='Path to data file')
    parser.add_argument('--traffic', action='store_true',
                    help='Enable traffic simulation')
    parser.add_argument('--driver_hours', action='store_true',
                    help='Enable driver hours constraints')
    parser.add_argument('--heterogeneous_fleet', action='store_true',
                    help='Use heterogeneous fleet')
    return parser.parse_args()

def run_command(cmd):
    """
    Run a command and return the process return code and output.
    
    Args:
        cmd: Command to run as a list of strings
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command directly without Popen to avoid I/O issues
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        # Print the output
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"Error (code {result.returncode}):")
            print(result.stderr)
            
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return 1, "", str(e)

def run_vrp_optimizer(args):
    """
    Run the VRP Optimizer with error handling.
    
    Args:
        args: Command-line arguments
    """
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Build the command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        "src/main.py",
        "--iterations", str(args.iterations),
        "--output", args.output,
    ]
    
    # Add optional arguments
    if args.config:
        cmd.extend(["--config", args.config])
    if args.data:
        cmd.extend(["--data", args.data])
    
    # Add optional flags
    if args.traffic:
        cmd.append("--traffic")
    if args.driver_hours:
        cmd.append("--driver_hours")
    if args.heterogeneous_fleet:
        cmd.append("--heterogeneous_fleet")
    
    # Run the command
    return_code, stdout, stderr = run_command(cmd)
    
    # If the command failed and we used traffic and driver_hours options
    if return_code != 0 and args.traffic and args.driver_hours:
        logger.warning("Command failed with traffic and driver_hours options. Retrying without these options.")
        
        # Remove the problematic options
        cmd_simplified = [
            sys.executable,
            "src/main.py",
            "--iterations", str(args.iterations),
            "--output", args.output,
        ]
        
        if args.config:
            cmd_simplified.extend(["--config", args.config])
        if args.data:
            cmd_simplified.extend(["--data", args.data])
        
        if args.heterogeneous_fleet:
            cmd_simplified.append("--heterogeneous_fleet")
        
        # Run the simplified command
        logger.info("Retrying with simplified options...")
        return_code, stdout, stderr = run_command(cmd_simplified)
    
    # Log completion time
    runtime = time.time() - start_time
    logger.info(f"Total runtime: {runtime:.2f} seconds")
    
    # Check if results were generated
    if os.path.exists(os.path.join(args.output, 'routes.html')):
        logger.info(f"Results saved to {args.output}/")
        print(f"\nResults successfully saved to {args.output}/")
    else:
        logger.error(f"No results were generated in {args.output}/")
        print(f"\nNo results were generated in {args.output}/")
    
    return return_code

if __name__ == '__main__':
    args = parse_args()
    sys.exit(run_vrp_optimizer(args)) 