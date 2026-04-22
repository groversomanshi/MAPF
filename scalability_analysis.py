import os
import time
import subprocess
import argparse
import yaml
import sys

def run_experiment(algo, input_file, output_file):
    """Runs the main.py script with the given algorithm and input file."""
    cmd = [
        "python3", "main.py",
        "--algo", algo,
        "--input", input_file,
        "--output", output_file
    ]
    # We do NOT include --visualize to keep the scalability test focused on planning time
    
    start_time = time.time()
    try:
        # Setting a timeout of 60 seconds per test to avoid hanging indefinitely
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60) 
        duration = time.time() - start_time
        success = "✓" in result.stdout
        return success, duration, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, 60.0, "Timeout (60s)"
    except Exception as e:
        return False, 0, str(e)

def ensure_env_exists(env_dir, count):
    """
    Checks if an environment with the required number of agents exists.
    If not, tries to create one by trimming a larger environment or generating one.
    """
    os.makedirs("tests/scalability_envs", exist_ok=True)
    target_path = f"tests/scalability_envs/env_agents_{count}.yaml"
    
    if os.path.exists(target_path):
        return target_path

    # Try to find an existing benchmark file to trim
    benchmark_files = []
    if os.path.exists(env_dir):
        benchmark_files = [os.path.join(env_dir, f) for f in os.listdir(env_dir) if f.endswith(".yaml")]
    
    for bf in benchmark_files:
        try:
            with open(bf, 'r') as f:
                data = yaml.safe_load(f)
            if len(data.get('agents', [])) >= count:
                # Trim and save
                data['agents'] = data['agents'][:count]
                with open(target_path, 'w') as f:
                    yaml.safe_dump(data, f)
                return target_path
        except:
            continue
            
    # If we get here, we might need to generate a 128-agent environment if 100 is the max
    # For now, let's assume we can at least find one to trim for 2, 4, 8, 16, 64.
    # If it's 128 and we only have 100, we'll return None and the script will skip.
    return None

def main():
    parser = argparse.ArgumentParser(description="Analyze algorithm scalability across robot counts.")
    parser.add_argument("--algo", required=True, help="Algorithm to test (cbs, mstar, mic, coupled, ma-cbs)")
    parser.add_argument("--env_dir", default="centralized/benchmark/32x32_obst204", help="Directory containing benchmark YAMLs")
    args = parser.parse_args()

    robot_counts = [2, 4, 8, 16, 64, 128]
    
    print(f"\n{'='*50}")
    print(f" SCALABILITY ANALYSIS: {args.algo.upper()}")
    print(f"{'='*50}")
    print(f"{'Robots':<10} | {'Status':<12} | {'Time (s)':<10} | {'Notes'}")
    print("-" * 60)

    for count in robot_counts:
        env_file = ensure_env_exists(args.env_dir, count)
        
        if env_file:
            out_file = f"tests/out_scalability_{args.algo}_{count}.yaml"
            success, duration, log = run_experiment(args.algo, env_file, out_file)
            
            status = "SUCCESS" if success else "FAILED"
            notes = ""
            if "Timeout" in log:
                notes = "Timed out (60s)"
            elif not success:
                notes = "No solution found"
                
            print(f"{count:<10} | {status:<12} | {duration:<10.2f} | {notes}")
        else:
            print(f"{count:<10} | {'SKIPPED':<12} | {'N/A':<10} | No env with {count} agents found")

    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
