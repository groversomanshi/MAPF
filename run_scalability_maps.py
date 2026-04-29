#!/usr/bin/env python3
"""Run map scalability experiments across agent counts and seeds."""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_AGENT_COUNTS = [2, 4, 8, 16, 64, 128]
DEFAULT_RUNS = 5


def parse_csv_ints(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def seed_label(seeds):
    if len(seeds) == 1:
        return f"seed{seeds[0]}"
    return "seeds" + "-".join(str(seed) for seed in seeds)


def run_command(cmd):
    print("\n$ " + " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_agent_count(args, agents):
    rows = []
    fieldnames = None

    with tempfile.TemporaryDirectory(prefix="parasol_scalability_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for seed in args.seeds:
            tmp_csv = tmp_dir / f"results_{args.algo}_agents{agents}_seed{seed}.csv"
            cmd = [
                sys.executable,
                "scalability_analysis.py",
                "maps",
                "--algo",
                args.algo,
                "--map_dir",
                args.map_dir,
                "--agents",
                str(agents),
                "--runs",
                str(args.runs),
                "--seed",
                str(seed),
                "--env_out_dir",
                args.env_out_dir,
                "--results_csv",
                str(tmp_csv),
            ]
            if args.maps:
                cmd.extend(["--maps", args.maps])

            run_command(cmd)
            seed_rows = read_csv_rows(tmp_csv)
            rows.extend(seed_rows)
            if fieldnames is None and seed_rows:
                fieldnames = list(seed_rows[0].keys())

    if fieldnames is None:
        raise RuntimeError(f"No rows were produced for agents={agents}")

    output_csv = args.output_dir / f"results_{args.algo}_agents{agents}_{seed_label(args.seeds)}.csv"
    write_csv(output_csv, rows, fieldnames)
    print(f"\nWrote merged results CSV: {output_csv} ({len(rows)} rows)")
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run scalability_analysis.py maps for multiple agent counts and seeds, "
            "then generate the combined visualization."
        )
    )
    parser.add_argument("algo", help="Algorithm to test, for example ma-cbs")
    parser.add_argument("seeds", nargs="+", type=int, help="One or more seeds, for example 0 4 123")
    parser.add_argument(
        "--agents",
        default=",".join(str(count) for count in DEFAULT_AGENT_COUNTS),
        help="Comma-separated agent counts to run",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Randomized runs per map per seed")
    parser.add_argument("--map-dir", default="main-maps", help="Directory containing map YAMLs")
    parser.add_argument("--maps", default="", help="Comma-separated map YAML filenames, default is all maps")
    parser.add_argument("--env-out-dir", default="tests/scalability_envs", help="Directory for generated env YAMLs")
    parser.add_argument("--output-dir", type=Path, default=Path("tests"), help="Directory for result CSVs and plot")
    parser.add_argument("--skip-plot", action="store_true", help="Only write result CSVs")
    args = parser.parse_args()

    args.agent_counts = parse_csv_ints(args.agents)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_csvs = [run_agent_count(args, agents) for agents in args.agent_counts]

    if args.skip_plot:
        return

    label = seed_label(args.seeds)
    summary_png = args.output_dir / f"results_{args.algo}_{label}_summary.png"
    summary_csv = args.output_dir / f"results_{args.algo}_{label}_summary.csv"
    run_command(
        [
            sys.executable,
            "tests/visualize_results.py",
            *[str(path) for path in result_csvs],
            "-o",
            str(summary_png),
            "--summary-csv",
            str(summary_csv),
        ]
    )


if __name__ == "__main__":
    main()
