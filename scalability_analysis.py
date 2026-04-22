import os
import time
import subprocess
import argparse
import yaml
import sys
import random
import csv
from collections import deque, defaultdict

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

def _safe_last_time(steps):
    if not steps:
        return 0
    last = steps[-1]
    if isinstance(last, dict):
        return int(last.get("t", 0))
    return int(getattr(last, "time", 0))

def _compute_schedule_makespan(schedule):
    if not isinstance(schedule, dict) or not schedule:
        return None
    return max(_safe_last_time(steps) for steps in schedule.values())

def _compute_schedule_sum_of_costs(schedule):
    if not isinstance(schedule, dict) or not schedule:
        return None
    return sum(_safe_last_time(steps) for steps in schedule.values())

def extract_output_metrics(output_file):
    """
    Extract metrics from an algorithm output YAML (when present).

    Returns a dict with keys: cost, makespan, planning_time_s, sum_of_costs, num_conflicts.
    Missing/unavailable metrics are returned as "" so they can be written to CSV.
    """
    if not output_file or not os.path.exists(output_file):
        return {
            "cost": "",
            "makespan": "",
            "planning_time_s": "",
            "sum_of_costs": "",
            "num_conflicts": "",
        }

    try:
        out = load_yaml(output_file) or {}
    except Exception:
        return {
            "cost": "",
            "makespan": "",
            "planning_time_s": "",
            "sum_of_costs": "",
            "num_conflicts": "",
        }

    schedule = out.get("schedule")
    makespan = out.get("makespan")
    if makespan is None:
        makespan = _compute_schedule_makespan(schedule)

    sum_of_costs = out.get("sum_of_costs")
    if sum_of_costs is None:
        sum_of_costs = _compute_schedule_sum_of_costs(schedule)

    planning_time = out.get("planning_time")
    if planning_time is None:
        planning_time_s = ""
    else:
        planning_time_s = f"{float(planning_time):.6f}"

    num_conflicts = out.get("num_conflicts", "")

    cost = out.get("cost", "")
    makespan = "" if makespan is None else makespan
    sum_of_costs = "" if sum_of_costs is None else sum_of_costs

    return {
        "cost": cost,
        "makespan": makespan,
        "planning_time_s": planning_time_s,
        "sum_of_costs": sum_of_costs,
        "num_conflicts": num_conflicts,
    }

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def dump_yaml(path, data):
    # NOTE: We intentionally use yaml.dump (not safe_dump) so that obstacle tuples
    # are preserved as !!python/tuple. Several solvers treat obstacles as tuples.
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

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
                data = yaml.load(f, Loader=yaml.FullLoader)
            if len(data.get('agents', [])) >= count:
                # Trim and save
                data['agents'] = data['agents'][:count]
                dump_yaml(target_path, data)
                return target_path
        except:
            continue
            
    # If we get here, we might need to generate a 128-agent environment if 100 is the max
    # For now, let's assume we can at least find one to trim for 2, 4, 8, 16, 64.
    # If it's 128 and we only have 100, we'll return None and the script will skip.
    return None

def _neighbors_4(x, y, width, height):
    if x > 0:
        yield (x - 1, y)
    if x + 1 < width:
        yield (x + 1, y)
    if y > 0:
        yield (x, y - 1)
    if y + 1 < height:
        yield (x, y + 1)

def compute_connected_components(dims, obstacles):
    width, height = dims[0], dims[1]
    obstacle_set = set(obstacles)
    comp_id_by_cell = {}
    comps = defaultdict(list)

    cid = 0
    for x in range(width):
        for y in range(height):
            cell = (x, y)
            if cell in obstacle_set or cell in comp_id_by_cell:
                continue
            q = deque([cell])
            comp_id_by_cell[cell] = cid
            while q:
                cx, cy = q.popleft()
                comps[cid].append((cx, cy))
                for nx, ny in _neighbors_4(cx, cy, width, height):
                    ncell = (nx, ny)
                    if ncell in obstacle_set or ncell in comp_id_by_cell:
                        continue
                    comp_id_by_cell[ncell] = cid
                    q.append(ncell)
            cid += 1

    return comps

def generate_random_env_from_map(map_yaml_path, agent_count, seed, run_index, out_dir="tests/scalability_envs"):
    os.makedirs(out_dir, exist_ok=True)
    map_data = load_yaml(map_yaml_path)
    if "map" not in map_data:
        raise ValueError(f"{map_yaml_path} is missing top-level 'map' key")

    dims = map_data["map"]["dimensions"]
    obstacles = map_data["map"].get("obstacles", [])
    comps = compute_connected_components(dims, obstacles)

    eligible = [cells for cells in comps.values() if len(cells) >= 2 * agent_count]
    if not eligible:
        raise ValueError(f"Map {map_yaml_path} has no connected component with >= {2*agent_count} free cells")

    rng = random.Random(f"{seed}:{os.path.basename(map_yaml_path)}:{run_index}:{agent_count}")
    component_cells = rng.choice(eligible)
    chosen = rng.sample(component_cells, 2 * agent_count)
    starts = chosen[:agent_count]
    goals = chosen[agent_count:]

    agents = []
    for i in range(agent_count):
        agents.append(
            {
                "name": f"agent{i}",
                "start": [int(starts[i][0]), int(starts[i][1])],
                "goal": [int(goals[i][0]), int(goals[i][1])],
            }
        )

    env = {"agents": agents, "map": map_data["map"]}
    map_stem = os.path.splitext(os.path.basename(map_yaml_path))[0]
    out_path = os.path.join(out_dir, f"env_{map_stem}_agents_{agent_count}_seed_{seed}_run_{run_index}.yaml")
    dump_yaml(out_path, env)
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Analyze algorithm scalability across robot counts and maps.")
    sub = parser.add_subparsers(dest="mode", required=True)

    bench = sub.add_parser("benchmark", help="Run scalability across agent counts by trimming benchmark environments.")
    bench.add_argument("--algo", required=True, help="Algorithm to test (cbs, mstar, mic, coupled, ma-cbs)")
    bench.add_argument("--env_dir", default="centralized/benchmark/32x32_obst204", help="Directory containing benchmark YAMLs")
    bench.add_argument("--robot_counts", default="2,4,8,16,64,128", help="Comma-separated robot counts to test")

    maps = sub.add_parser("maps", help="Run a fixed agent count on one or more maps with randomized starts/goals.")
    maps.add_argument("--algo", required=True, help="Algorithm to test (cbs, mstar, mic, coupled, ma-cbs)")
    maps.add_argument("--map_dir", default="main-maps", help="Directory containing map YAMLs (map only, no agents)")
    maps.add_argument("--maps", default="", help="Comma-separated map YAML filenames to use (default: all in map_dir)")
    maps.add_argument("--agents", type=int, required=True, help="Number of agents for each run")
    maps.add_argument("--runs", type=int, default=1, help="Number of randomized runs per map")
    maps.add_argument("--seed", type=int, default=0, help="Seed for deterministic randomization")
    maps.add_argument("--env_out_dir", default="tests/scalability_envs", help="Where to write generated env YAMLs")
    maps.add_argument("--results_csv", default="", help="Optional CSV path to write a per-run results table")

    args = parser.parse_args()

    if args.mode == "benchmark":
        robot_counts = [int(x) for x in args.robot_counts.split(",") if x.strip()]

        print(f"\n{'='*50}")
        print(f" SCALABILITY (BENCHMARK): {args.algo.upper()}")
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
        return

    if args.mode == "maps":
        if args.maps.strip():
            map_files = [os.path.join(args.map_dir, m.strip()) for m in args.maps.split(",") if m.strip()]
        else:
            map_files = [
                os.path.join(args.map_dir, f)
                for f in sorted(os.listdir(args.map_dir))
                if f.endswith(".yaml") or f.endswith(".yml")
            ]

        print(f"\n{'='*80}")
        print(f" MAP RUNS: {args.algo.upper()} | agents={args.agents} | runs/map={args.runs} | seed={args.seed}")
        print(f"{'='*80}")
        print(f"{'Map':<18} | {'Run':<4} | {'Status':<12} | {'Time (s)':<10} | {'Notes'}")
        print("-" * 80)

        summary = defaultdict(lambda: {"ok": 0, "total": 0, "times": []})
        rows = []

        for map_path in map_files:
            map_name = os.path.splitext(os.path.basename(map_path))[0]
            for run_idx in range(args.runs):
                try:
                    env_file = generate_random_env_from_map(
                        map_path,
                        agent_count=args.agents,
                        seed=args.seed,
                        run_index=run_idx,
                        out_dir=args.env_out_dir,
                    )
                except Exception as e:
                    print(f"{map_name:<18} | {run_idx:<4} | {'SKIPPED':<12} | {'N/A':<10} | {e}")
                    rows.append(
                        {
                            "map": map_name,
                            "run": run_idx,
                            "algo": args.algo,
                            "agents": args.agents,
                            "seed": args.seed,
                            "status": "SKIPPED",
                            "time_s": "",
                            "notes": str(e),
                            "env_file": "",
                            "output_file": "",
                        }
                    )
                    continue

                out_file = f"tests/out_maps_{args.algo}_{map_name}_agents_{args.agents}_run_{run_idx}.yaml"
                success, duration, log = run_experiment(args.algo, env_file, out_file)

                status = "SUCCESS" if success else "FAILED"
                notes = ""
                if "Timeout" in log:
                    notes = "Timed out (60s)"
                elif not success:
                    notes = "No solution found"

                metrics = extract_output_metrics(out_file) if success else {
                    "cost": "",
                    "makespan": "",
                    "planning_time_s": "",
                    "sum_of_costs": "",
                    "num_conflicts": "",
                }

                print(f"{map_name:<18} | {run_idx:<4} | {status:<12} | {duration:<10.2f} | {notes}")
                rows.append(
                    {
                        "map": map_name,
                        "run": run_idx,
                        "algo": args.algo,
                        "agents": args.agents,
                        "seed": args.seed,
                        "status": status,
                        "time_s": f"{duration:.4f}",
                        "planning_time_s": metrics["planning_time_s"],
                        "makespan": metrics["makespan"],
                        "cost": metrics["cost"],
                        "sum_of_costs": metrics["sum_of_costs"],
                        "num_conflicts": metrics["num_conflicts"],
                        "notes": notes,
                        "env_file": env_file,
                        "output_file": out_file,
                    }
                )

                summary[map_name]["total"] += 1
                if success:
                    summary[map_name]["ok"] += 1
                    summary[map_name]["times"].append(duration)

        print("-" * 80)
        print("Summary (successful runs only):")
        for map_name in sorted(summary.keys()):
            s = summary[map_name]
            if s["ok"] == 0:
                print(f"- {map_name}: 0/{s['total']} succeeded")
            else:
                avg = sum(s["times"]) / len(s["times"])
                print(f"- {map_name}: {s['ok']}/{s['total']} succeeded, avg time {avg:.2f}s")
        print(f"{'='*80}\n")

        if args.results_csv.strip():
            os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
            fieldnames = [
                "map",
                "run",
                "algo",
                "agents",
                "seed",
                "status",
                "time_s",
                "planning_time_s",
                "makespan",
                "cost",
                "sum_of_costs",
                "num_conflicts",
                "notes",
                "env_file",
                "output_file",
            ]
            with open(args.results_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"✓ Wrote results CSV to {args.results_csv}\n")

        return

if __name__ == "__main__":
    main()
