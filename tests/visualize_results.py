#!/usr/bin/env python3
"""Create summary plots from scalability results CSV files."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


MAP_COLORS = {
    "open": "#2b8cbe",
    "narrow-passages": "#7b3294",
    "cluttered": "#d95f0e",
}


def parse_float(value):
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def parse_int(value):
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else None


def load_rows(csv_paths):
    rows = []
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["source_csv"] = str(csv_path)
                row["agents"] = parse_int(row.get("agents"))
                row["run"] = parse_int(row.get("run"))
                row["seed"] = parse_int(row.get("seed"))
                row["time_s"] = parse_float(row.get("time_s"))
                row["planning_time_s"] = parse_float(row.get("planning_time_s"))
                row["makespan"] = parse_float(row.get("makespan"))
                row["cost"] = parse_float(row.get("cost"))
                row["sum_of_costs"] = parse_float(row.get("sum_of_costs"))
                row["num_conflicts"] = parse_float(row.get("num_conflicts"))
                row["success"] = row.get("status") == "SUCCESS"
                rows.append(row)
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["map"], row["agents"])].append(row)

    summaries = []
    for (map_name, agents), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        successes = [row for row in group if row["success"]]
        success_rate = len(successes) / len(group) if group else 0

        def values(field, success_only=True):
            source = successes if success_only else group
            return [row[field] for row in source if row.get(field) is not None]

        summaries.append(
            {
                "map": map_name,
                "agents": agents,
                "runs": len(group),
                "successes": len(successes),
                "success_rate": success_rate,
                "median_time_s_all": median(values("time_s", success_only=False)) if values("time_s", False) else None,
                "mean_time_s_all": mean(values("time_s", success_only=False)) if values("time_s", False) else None,
                "median_time_s_success": median(values("time_s")) if values("time_s") else None,
                "median_planning_time_s": median(values("planning_time_s")) if values("planning_time_s") else None,
                "median_makespan": median(values("makespan")) if values("makespan") else None,
                "median_cost": median(values("cost")) if values("cost") else None,
                "mean_num_conflicts": mean(values("num_conflicts")) if values("num_conflicts") else None,
            }
        )
    return summaries


def write_summary_csv(summaries, output_path):
    fieldnames = [
        "map",
        "agents",
        "runs",
        "successes",
        "success_rate",
        "median_time_s_all",
        "mean_time_s_all",
        "median_time_s_success",
        "median_planning_time_s",
        "median_makespan",
        "median_cost",
        "mean_num_conflicts",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def plot_metric(ax, summaries, map_names, metric, title, ylabel, *, percent=False, log_y=False):
    for map_name in map_names:
        points = [summary for summary in summaries if summary["map"] == map_name]
        points = sorted(points, key=lambda summary: summary["agents"])
        xs = [summary["agents"] for summary in points]
        ys = [summary[metric] for summary in points]
        plot_xs = [x for x, y in zip(xs, ys) if y is not None and (not log_y or y > 0)]
        plot_ys = [y for y in ys if y is not None and (not log_y or y > 0)]
        if percent:
            plot_ys = [y * 100 for y in plot_ys]
        if not plot_xs:
            continue

        ax.plot(
            plot_xs,
            plot_ys,
            marker="o",
            linewidth=2,
            markersize=5,
            label=map_name,
            color=MAP_COLORS.get(map_name),
        )

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Agents")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    if percent:
        ax.set_ylim(-4, 104)


def annotate_success_counts(ax, summaries):
    for summary in summaries:
        rate = summary["success_rate"] * 100
        ax.annotate(
            f"{summary['successes']}/{summary['runs']}",
            (summary["agents"], rate),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            alpha=0.75,
        )


def make_plot(summaries, output_path):
    map_names = sorted({summary["map"] for summary in summaries})

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("MA-CBS Scalability Results by Map Type", fontsize=16, weight="bold")

    plot_metric(
        axes[0][0],
        summaries,
        map_names,
        "success_rate",
        "Success Rate",
        "Successful runs (%)",
        percent=True,
    )
    annotate_success_counts(axes[0][0], summaries)

    plot_metric(
        axes[0][1],
        summaries,
        map_names,
        "median_time_s_all",
        "Median Wall Time, All Runs",
        "Seconds, failures include timeout",
        log_y=True,
    )
    plot_metric(
        axes[0][2],
        summaries,
        map_names,
        "median_planning_time_s",
        "Median Planning Time, Successes",
        "Seconds",
        log_y=True,
    )
    plot_metric(
        axes[1][0],
        summaries,
        map_names,
        "median_cost",
        "Median Cost, Successes",
        "Cost",
    )
    plot_metric(
        axes[1][1],
        summaries,
        map_names,
        "median_makespan",
        "Median Makespan, Successes",
        "Timesteps",
    )
    plot_metric(
        axes[1][2],
        summaries,
        map_names,
        "mean_num_conflicts",
        "Mean Conflicts, Successes",
        "Conflicts",
    )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=max(1, len(labels)), frameon=False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def print_summary(summaries):
    print("Summary by map and agent count:")
    for summary in summaries:
        success_pct = summary["success_rate"] * 100
        wall = summary["median_time_s_all"]
        plan = summary["median_planning_time_s"]
        wall_text = f"{wall:.3g}s" if wall is not None and math.isfinite(wall) else "n/a"
        plan_text = f"{plan:.3g}s" if plan is not None and math.isfinite(plan) else "n/a"
        print(
            f"- {summary['map']:15s} agents={summary['agents']:3d} "
            f"success={summary['successes']}/{summary['runs']} ({success_pct:5.1f}%) "
            f"median_wall={wall_text:>7s} median_plan_success={plan_text:>7s}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MA-CBS scalability result CSVs across map types and agent counts."
    )
    parser.add_argument("csvs", nargs="+", type=Path, help="Result CSV files to combine")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("tests/results_ma-cbs_summary.png"),
        help="Path for the output PNG plot",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to write the aggregated summary table",
    )
    args = parser.parse_args()

    rows = load_rows(args.csvs)
    if not rows:
        raise SystemExit("No result rows found.")

    summaries = summarize(rows)
    print_summary(summaries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_plot(summaries, args.output)
    print(f"\nSaved plot to {args.output}")

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        write_summary_csv(summaries, args.summary_csv)
        print(f"Saved summary CSV to {args.summary_csv}")


if __name__ == "__main__":
    main()
