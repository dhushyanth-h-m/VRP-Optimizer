#!/usr/bin/env python3
"""
Simple VRP Optimizer Runner - Runs the basic optimization without problematic options
"""

import os
import subprocess
import sys
import time

def main():
    # Create output directory
    output_dir = "results_simple"
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic command - only use options known to work
    cmd = [
        sys.executable,
        "src/main.py",
        "--iterations", "15",
        "--output", output_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    # Run the command directly
    try:
        result = subprocess.run(cmd, check=False, text=True)
        
        # Log completion
        runtime = time.time() - start_time
        print(f"Command completed in {runtime:.2f} seconds with exit code {result.returncode}")
        
        # Check if results were generated
        if os.path.exists(os.path.join(output_dir, 'routes.html')):
            print(f"Results successfully saved to {output_dir}/")
        else:
            print(f"No results were generated in {output_dir}/")
            
    except Exception as e:
        print(f"Error executing command: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 