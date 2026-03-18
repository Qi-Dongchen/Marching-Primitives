"""
Visualize superquadric primitives from:
    data/chair*/chair*_normalized_sq.csv

For each CSV, this script creates a PNG rendering of all primitives.
"""

from __future__ import annotations

import argparse
import glob
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from marching_primitives.single_mesh_superquadrics import single_mesh_superquadrics


def load_sq_csv(csv_path: Path) -> np.ndarray:
    """Load one superquadric CSV into a (K, 11) float array."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 11:
        raise ValueError(
            f"{csv_path} has {data.shape[1]} columns; expected 11 SQ parameters."
        )
    return data


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    """Set equal axis scaling around all plotted points."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * np.max(maxs - mins)
    radius = max(radius, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_one_csv(csv_path: Path, output_png: Path, arclength: float, dpi: int) -> int:
    """Render all superquadrics from one CSV and save to PNG."""
    params = load_sq_csv(csv_path)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")

    all_points = []
    for i, sq in enumerate(params):
        x_mesh, y_mesh, z_mesh = single_mesh_superquadrics(
            sq, arclength=arclength, taper=False
        )
        ax.plot_surface(
            x_mesh,
            y_mesh,
            z_mesh,
            color=cmap(i % 20),
            linewidth=0,
            antialiased=True,
            alpha=0.9,
            shade=True,
        )
        all_points.append(
            np.column_stack((x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()))
        )

    if all_points:
        set_equal_axes(ax, np.vstack(all_points))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{csv_path.stem}  |  {params.shape[0]} primitives")
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)
    return params.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize superquadrics from chair*_normalized_sq.csv files."
    )
    parser.add_argument(
        "--pattern",
        default="data/chair*/chair*_normalized_sq.csv",
        help="Glob pattern for SQ CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sq_visualizations",
        help="Directory to save generated PNG files.",
    )
    parser.add_argument(
        "--arclength",
        type=float,
        default=0.02,
        help="Sampling arclength for superquadric meshing.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="DPI for saved images.",
    )
    args = parser.parse_args()

    csv_files = sorted(Path(p) for p in glob.glob(args.pattern))
    if not csv_files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    output_dir = Path(args.output_dir)
    print(f"Found {len(csv_files)} SQ CSV files.")

    for csv_path in csv_files:
        output_png = output_dir / f"{csv_path.stem}.png"
        count = render_one_csv(csv_path, output_png, args.arclength, args.dpi)
        print(f"[OK] {csv_path} -> {output_png} ({count} primitives)")


if __name__ == "__main__":
    main()
