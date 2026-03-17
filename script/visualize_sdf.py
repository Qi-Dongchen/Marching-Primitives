"""
Visualize a Signed Distance Field (SDF) from a CSV file.

Usage:
    python visualize_sdf.py <path_to_sdf.csv> [options]

Options:
    --threshold FLOAT   SDF threshold for iso-surface display (default: 0.0)
    --mode MODE         Visualization mode: scatter, isosurface, slice (default: scatter)
    --slice-axis AXIS   Axis for slice mode: x, y, z (default: z)
    --slice-index INT   Slice index along the chosen axis (default: middle)
    --alpha FLOAT       Point/surface opacity (default: 0.3)
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_sdf(csv_file):
    """Load SDF from CSV and build voxel grid."""
    raw = np.loadtxt(csv_file, delimiter=',').flatten()

    grid_size = int(raw[0])
    voxel_grid = {
        'size': np.array([grid_size, grid_size, grid_size]),
        'range': raw[1:7],
    }
    sdf = raw[7:]

    voxel_grid['x'] = np.linspace(voxel_grid['range'][0], voxel_grid['range'][1], grid_size)
    voxel_grid['y'] = np.linspace(voxel_grid['range'][2], voxel_grid['range'][3], grid_size)
    voxel_grid['z'] = np.linspace(voxel_grid['range'][4], voxel_grid['range'][5], grid_size)

    voxel_grid['interval'] = (voxel_grid['range'][1] - voxel_grid['range'][0]) / (grid_size - 1)
    voxel_grid['truncation'] = 1.2 * voxel_grid['interval']

    return sdf, voxel_grid


def visualize_scatter(sdf, grid, threshold, alpha):
    """3D scatter plot of SDF points near the surface."""
    sdf3d = sdf.reshape(grid['size'], order='F')
    gx, gy, gz = np.meshgrid(grid['x'], grid['y'], grid['z'], indexing='ij')

    mask = np.abs(sdf3d) <= threshold if threshold > 0 else sdf3d <= 0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    pts_x = gx[mask]
    pts_y = gy[mask]
    pts_z = gz[mask]
    vals = sdf3d[mask]

    sc = ax.scatter(pts_x, pts_y, pts_z, c=vals, cmap='coolwarm',
                    alpha=alpha, s=1, vmin=-grid['truncation'], vmax=grid['truncation'])
    plt.colorbar(sc, ax=ax, label='SDF value', shrink=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_title(f'SDF Scatter (threshold={threshold:.4f})')
    return fig


def visualize_isosurface(sdf, grid, threshold, alpha):
    """Marching-cubes iso-surface of the SDF."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("Error: scikit-image is required for isosurface mode.")
        print("Install with: pip install scikit-image")
        sys.exit(1)

    sdf3d = sdf.reshape(grid['size'], order='F')
    spacing = (
        (grid['range'][1] - grid['range'][0]) / (grid['size'][0] - 1),
        (grid['range'][3] - grid['range'][2]) / (grid['size'][1] - 1),
        (grid['range'][5] - grid['range'][4]) / (grid['size'][2] - 1),
    )

    verts, faces, _, _ = marching_cubes(sdf3d, level=threshold, spacing=spacing)
    # Offset vertices to match grid origin
    verts[:, 0] += grid['range'][0]
    verts[:, 1] += grid['range'][2]
    verts[:, 2] += grid['range'][4]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=alpha, edgecolor='k', linewidth=0.1)
    mesh.set_facecolor([0.56, 0.64, 0.69])
    ax.add_collection3d(mesh)

    ax.set_xlim(grid['range'][0], grid['range'][1])
    ax.set_ylim(grid['range'][2], grid['range'][3])
    ax.set_zlim(grid['range'][4], grid['range'][5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_title(f'SDF Iso-surface (level={threshold:.4f})')
    return fig


def visualize_slices(sdf, grid, slice_axis, slice_index):
    """2D heatmap slices of the SDF."""
    sdf3d = sdf.reshape(grid['size'], order='F')

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax_idx = axis_map[slice_axis]
    max_idx = grid['size'][ax_idx]

    if slice_index is None:
        slice_index = max_idx // 2

    slice_index = np.clip(slice_index, 0, max_idx - 1)

    if ax_idx == 0:
        data = sdf3d[slice_index, :, :]
        extent = [grid['range'][2], grid['range'][3], grid['range'][4], grid['range'][5]]
        xlabel, ylabel = 'Y', 'Z'
        slice_coord = grid['x'][slice_index]
    elif ax_idx == 1:
        data = sdf3d[:, slice_index, :]
        extent = [grid['range'][0], grid['range'][1], grid['range'][4], grid['range'][5]]
        xlabel, ylabel = 'X', 'Z'
        slice_coord = grid['y'][slice_index]
    else:
        data = sdf3d[:, :, slice_index]
        extent = [grid['range'][0], grid['range'][1], grid['range'][2], grid['range'][3]]
        xlabel, ylabel = 'X', 'Y'
        slice_coord = grid['z'][slice_index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    im = axes[0].imshow(data.T, origin='lower', extent=extent, cmap='coolwarm',
                        vmin=-grid['truncation'], vmax=grid['truncation'], aspect='equal')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f'SDF Heatmap ({slice_axis.upper()}={slice_coord:.4f}, idx={slice_index})')
    plt.colorbar(im, ax=axes[0], label='SDF value')

    # Contour with zero level highlighted
    cs = axes[1].contourf(data.T, levels=30, extent=extent, cmap='coolwarm', origin='lower')
    axes[1].contour(data.T, levels=[0], extent=extent, colors='black', linewidths=2, origin='lower')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f'SDF Contour (zero-level in black)')
    axes[1].set_aspect('equal')
    plt.colorbar(cs, ax=axes[1], label='SDF value')

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize SDF from CSV')
    parser.add_argument('csv_file', help='Path to the SDF .csv file')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='SDF threshold (scatter: show |sdf|<=t; isosurface: level)')
    parser.add_argument('--mode', choices=['scatter', 'isosurface', 'slice', 'all'],
                        default='all', help='Visualization mode')
    parser.add_argument('--slice-axis', choices=['x', 'y', 'z'], default='z',
                        help='Axis for slice mode')
    parser.add_argument('--slice-index', type=int, default=None,
                        help='Slice index (default: middle)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Opacity for scatter/isosurface')
    args = parser.parse_args()

    print(f"Loading SDF from {args.csv_file}...")
    sdf, grid = load_sdf(args.csv_file)
    print(f"Grid size: {grid['size']}, truncation: {grid['truncation']:.4f}")
    print(f"SDF range: [{np.nanmin(sdf):.4f}, {np.nanmax(sdf):.4f}]")

    threshold = args.threshold if args.threshold > 0 else grid['truncation']

    if args.mode in ('scatter', 'all'):
        print("Generating scatter plot...")
        visualize_scatter(sdf, grid, threshold, args.alpha)

    if args.mode in ('isosurface', 'all'):
        print("Generating iso-surface...")
        visualize_isosurface(sdf, grid, 0.0, args.alpha)

    if args.mode in ('slice', 'all'):
        print("Generating slice view...")
        visualize_slices(sdf, grid, args.slice_axis, args.slice_index)

    plt.show()


if __name__ == '__main__':
    main()
