#!/usr/bin/env python3
"""Visualize only the static map from a Parasol YAML environment file."""

# TO RUN: 
# python3 main-maps/visualize_map.py main-maps/cluttered.yaml
# python3 main-maps/visualize_map.py main-maps/cluttered.yaml -o cluttered_map.png

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml


def load_map(path):
    with open(path, "r", encoding="utf-8") as map_file:
        data = yaml.load(map_file, Loader=yaml.FullLoader)

    if not isinstance(data, dict) or "map" not in data:
        raise ValueError(f"{path} does not contain a top-level 'map' section")

    map_data = data["map"]
    if "dimensions" not in map_data:
        raise ValueError(f"{path} is missing map.dimensions")

    dimensions = map_data["dimensions"]
    if len(dimensions) != 2:
        raise ValueError(f"{path} map.dimensions must be [width, height]")

    obstacles = [tuple(obstacle) for obstacle in map_data.get("obstacles", [])]
    return (int(dimensions[0]), int(dimensions[1])), obstacles


def draw_map(dimensions, obstacles, title, show_grid=True, show_labels=True):
    width, height = dimensions
    obstacle_set = set(obstacles)
    obstacle_count = len(obstacle_set)
    density = obstacle_count / (width * height) if width and height else 0

    aspect = width / height if height else 1
    fig_width = min(max(width / 4, 6), 14)
    fig_height = fig_width / aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f7f4")

    for x, y in obstacle_set:
        if 0 <= x < width and 0 <= y < height:
            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor="#d34a4a",
                    edgecolor="#9f2f2f",
                    linewidth=0.8,
                )
            )
        else:
            print(f"Warning: obstacle {(x, y)} is outside dimensions {dimensions}")

    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            width,
            height,
            fill=False,
            edgecolor="#333333",
            linewidth=1.5,
        )
    )

    ax.set_title(
        f"{title}\n{width}x{height}, {obstacle_count} obstacles, {density:.1%} blocked",
        pad=12,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks(range(width))
    ax.set_yticks(range(height))

    if not show_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if show_grid:
        ax.grid(which="major", color="#d8d8d8", linewidth=0.6)
    else:
        ax.grid(False)

    ax.tick_params(axis="both", which="both", length=0)
    fig.tight_layout()
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Visualize only the static map obstacles from a YAML map/environment file."
    )
    parser.add_argument("map_file", help="Path to a YAML file with map.dimensions and map.obstacles")
    parser.add_argument(
        "-o",
        "--output",
        help="Save the map image to this file instead of opening an interactive window",
    )
    parser.add_argument("--no-grid", action="store_true", help="Hide grid lines")
    parser.add_argument("--no-labels", action="store_true", help="Hide x/y tick labels")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI when saving")
    args = parser.parse_args()

    path = Path(args.map_file)
    dimensions, obstacles = load_map(path)
    fig, _ = draw_map(
        dimensions,
        obstacles,
        title=path.name,
        show_grid=not args.no_grid,
        show_labels=not args.no_labels,
    )

    width, height = dimensions
    density = len(set(obstacles)) / (width * height)
    print(f"{path}: {width}x{height}, {len(set(obstacles))} obstacles, {density:.1%} blocked")

    if args.output:
        output_path = Path(args.output)
        fig.savefig(output_path, dpi=args.dpi)
        print(f"Saved map image to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
