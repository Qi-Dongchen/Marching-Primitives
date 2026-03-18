"""
Compare extracted superquadric primitives against a real scan.

Loads superquadric parameters from CSV (one row per primitive, 11 columns)
and optionally a ground-truth mesh (PLY/STL) or SDF (CSV) for comparison.

Usage:
    python compare_sq.py <sq_params.csv> [--scan <mesh.ply|sdf.csv>] [--arclength 0.02]

Examples:
    # Visualize primitives only (colored per primitive)
    python compare_sq.py data/chair1_normalized_sq.csv

    # Overlay with ground-truth PLY mesh
    python compare_sq.py data/chair1_normalized_sq.csv --scan data/chair1_normalized_watertight.ply

    # Overlay with original SDF
    python compare_sq.py data/chair1_normalized_sq.csv --scan data/chair1_normalized.csv
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from marching_primitives.single_mesh_superquadrics import single_mesh_superquadrics
from marching_primitives.mesh2tri import mesh2tri
from marching_primitives.sdf_superquadric import sdf_superquadric


def load_sq_params(csv_file):
    """Load superquadric parameters: (K, 11) array."""
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_scan(scan_path):
    """Load a scan file. Returns (vertices, faces) or (sdf, grid)."""
    ext = os.path.splitext(scan_path)[1].lower()

    if ext == '.ply':
        from marching_primitives.plyread import plyread
        faces, verts = plyread(scan_path, 'tri')
        return 'mesh', (verts, faces)

    elif ext == '.stl':
        try:
            from stl import mesh as stl_mesh
            m = stl_mesh.Mesh.from_file(scan_path)
            verts = m.vectors.reshape(-1, 3)
            faces = np.arange(len(verts)).reshape(-1, 3)
            return 'mesh', (verts, faces)
        except ImportError:
            print("Warning: numpy-stl not installed, trying manual STL read")
            return None, None

    elif ext == '.csv':
        raw = np.loadtxt(scan_path, delimiter=',').flatten()
        grid_size = int(raw[0])
        grid = {
            'size': np.array([grid_size, grid_size, grid_size]),
            'range': raw[1:7],
        }
        grid['x'] = np.linspace(grid['range'][0], grid['range'][1], grid_size)
        grid['y'] = np.linspace(grid['range'][2], grid['range'][3], grid_size)
        grid['z'] = np.linspace(grid['range'][4], grid['range'][5], grid_size)
        grid['interval'] = (grid['range'][1] - grid['range'][0]) / (grid_size - 1)
        grid['truncation'] = 1.2 * grid['interval']
        sdf = raw[7:]
        return 'sdf', (sdf, grid)

    else:
        print(f"Unsupported scan format: {ext}")
        return None, None


def extract_isosurface(sdf, grid, level=0.0):
    """Extract iso-surface vertices and faces from SDF."""
    from skimage.measure import marching_cubes
    sdf3d = sdf.reshape(grid['size'], order='F')
    spacing = tuple(
        (grid['range'][2*i+1] - grid['range'][2*i]) / (grid['size'][i] - 1)
        for i in range(3)
    )
    verts, faces, _, _ = marching_cubes(sdf3d, level=level, spacing=spacing)
    verts[:, 0] += grid['range'][0]
    verts[:, 1] += grid['range'][2]
    verts[:, 2] += grid['range'][4]
    return verts, faces


def main():
    parser = argparse.ArgumentParser(description='Compare superquadrics vs scan')
    parser.add_argument('sq_csv', help='Path to superquadric parameters CSV')
    parser.add_argument('--scan', help='Path to scan file (PLY, STL, or SDF CSV)',
                        default=None)
    parser.add_argument('--arclength', type=float, default=0.02,
                        help='Mesh sampling arclength')
    args = parser.parse_args()

    # Load superquadric parameters
    x = load_sq_params(args.sq_csv)
    print(f"Loaded {x.shape[0]} superquadrics from {args.sq_csv}")

    # Mesh each superquadric
    cmap = plt.cm.get_cmap('tab20', x.shape[0])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    all_verts = []
    for i in range(x.shape[0]):
        x_mesh, y_mesh, z_mesh = single_mesh_superquadrics(
            x[i], arclength=args.arclength, taper=False
        )
        f_i, v_i = mesh2tri(x_mesh, y_mesh, z_mesh, 'f')
        s = f_i.shape[0]
        idx1 = np.arange(s // 8, 3 * s // 8)
        idx2 = np.arange(5 * s // 8, 7 * s // 8)
        f_i = f_i[np.concatenate([idx1, idx2])]

        ax.plot_trisurf(v_i[:, 0], v_i[:, 1], v_i[:, 2],
                        triangles=f_i, color=cmap(i), edgecolor='none',
                        alpha=0.8, label=f'SQ {i}')
        all_verts.append(v_i)

    # Overlay scan if provided
    if args.scan:
        scan_type, scan_data = load_scan(args.scan)

        if scan_type == 'mesh':
            verts, faces = scan_data
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=faces, color='gray', edgecolor='none',
                            alpha=0.3, label='Scan')
            print(f"Overlaid mesh scan: {verts.shape[0]} vertices")

        elif scan_type == 'sdf':
            sdf, grid = scan_data
            try:
                verts, faces = extract_isosurface(sdf, grid)
                ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                                triangles=faces, color='gray', edgecolor='none',
                                alpha=0.3, label='Scan')
                print(f"Overlaid SDF iso-surface: {verts.shape[0]} vertices")
            except Exception as e:
                print(f"Could not extract iso-surface: {e}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_title(f'{x.shape[0]} Superquadric Primitives')

    # Auto-fit axis limits from all vertices
    if all_verts:
        all_v = np.vstack(all_verts)
        margin = 0.05
        for setter, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            lo, hi = all_v[:, dim].min(), all_v[:, dim].max()
            pad = (hi - lo) * margin
            setter(lo - pad, hi + pad)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
