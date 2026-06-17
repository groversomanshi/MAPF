import argparse
import importlib
import time
import yaml
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Path Planning Algorithms")
    parser.add_argument("--algo", choices=["ma-cbs", "cbs", "mic", "mstar", "coupled", "pibt"], required=True)
    parser.add_argument("--input", required=True, help="Input YAML file")
    parser.add_argument("--output", required=True, help="Output YAML file")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization video")
    
    args = parser.parse_args()

    # Load parameters
    try:
        with open(args.input, "r") as param_file:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    
    # Pre-process agents to ignore any malformed entries (e.g. from commented yaml)
    raw_agents = param.get("agents", [])
    agents = []
    for a in raw_agents:
        if isinstance(a, dict) and "name" in a and "start" in a and "goal" in a:
            agents.append(a)
        else:
            print(f"Warning: Ignored malformed agent config: {a}")

    solution = None
    output_data = {}

    if args.algo == "ma-cbs":
        mod = importlib.import_module("centralized.ma-cbs.ma-cbs")
        MACBSEnvironment = mod.MACBSEnvironment
        MACBS = mod.MACBS
        env = MACBSEnvironment(dimension, agents, obstacles)
        solver = MACBS(env, param.get("merge_bound", 1))
        planning_start = time.perf_counter()
        solution = solver.search()
        planning_time = time.perf_counter() - planning_start
        if solution:
            output_data = mod.build_output(
                solution,
                env,
                planning_time,
                solver.num_conflicts,
            )
            
    elif args.algo == "cbs":
        mod = importlib.import_module("centralized.cbs.cbs")
        Environment = mod.Environment
        CBS = mod.CBS
        env = Environment(dimension, agents, obstacles)
        cbs = CBS(env)
        solution = cbs.search()
        if solution:
            cost = env.compute_solution_cost(solution)
            output_data = {"schedule": solution, "cost": cost}
            
    elif args.algo == "mic":
        mod = importlib.import_module("centralized.mic.mic")
        Environment = mod.Environment
        MiC = mod.MiC
        env = Environment(dimension, agents, obstacles)
        solver = MiC(env)
        paths = solver.search()
        if paths:
            solution = solver.build_plan(paths)
            cost = env.compute_solution_cost(paths)
            output_data = {"schedule": solution, "cost": cost}
            
    elif args.algo == "mstar":
        mod = importlib.import_module("centralized.mstar.mstar")
        Environment = mod.Environment
        MStar = mod.MStar
        env = Environment(dimension, agents, obstacles)
        solver = MStar(env)
        solution = solver.search()
        if solution:
            cost = env.compute_solution_cost(solution)
            output_data = {"schedule": solution, "cost": cost}
            
    elif args.algo == "coupled":
        mod = importlib.import_module("coupled.coupled")
        Environment = mod.Environment
        CoupledPlanner = mod.CoupledPlanner
        env = Environment(dimension, agents, obstacles)
        planner = CoupledPlanner(env)
        raw_solution = planner.solve()
        
        if raw_solution:
            solution = True
            cost = len(raw_solution)
            schedule = {}
            for name in env.get_agent_names():
                schedule[name] = []
            for t, js in enumerate(raw_solution):
                for i, name in enumerate(env.get_agent_names()):
                    loc = js.locations[i]
                    schedule[name].append({"t": t, "x": loc.x, "y": loc.y})
            output_data = {"schedule": schedule, "cost": cost}

    elif args.algo == "pibt":
        mod = importlib.import_module("decentralized.pibt.pibt")
        Environment = mod.Environment
        PIBT = mod.PIBT
        env = Environment(dimension, agents, obstacles)
        solver = PIBT(env)
        solution = solver.search(param.get("max_timesteps"))
        if solution:
            cost = env.compute_solution_cost(solution)
            output_data = {"schedule": solution, "cost": cost}

    if not solution:
        print(f"✗ No solution found using {args.algo}")
        return

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        yaml.safe_dump(output_data, f)
    print(f"✓ Solution successfully written to {args.output} (Cost: {output_data['cost']})")

    if args.visualize:
        from utils.visualize import Animation
        print(f"Generating visualization...")
        anim = Animation(param, output_data)
        video_path = args.output.replace('.yaml', '.gif').replace('.yml', '.gif')
        if video_path == args.output:
            video_path = args.output + ".gif"
        anim.save(video_path, 1.0)
        print(f"✓ Video saved to {video_path}")

if __name__ == "__main__":
    main()
