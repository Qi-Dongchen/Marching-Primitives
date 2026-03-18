"""
CLI entry point for Marching-Primitives algorithm.

Loads an SDF from CSV, runs MPS to extract superquadrics,
generates mesh, and visualizes the results.

Usage (after pip install):
    marching-primitives <path_to_sdf.csv> [--ply <path_to_watertight.ply>]
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .mps import MPS
from .mesh_superquadrics import mesh_superquadrics
from .mesh2tri import mesh2tri
from .single_mesh_superquadrics import single_mesh_superquadrics
from .plyread import plyread


def reduce_mesh(faces, vertices, ratio=0.5):
    """
    Simple mesh decimation by random face removal.

    For production use, consider using PyMeshLab or Open3D for proper
    mesh decimation (equivalent to MATLAB's reducepatch).

    Parameters
    ----------
    faces : ndarray (M, 3)
    vertices : ndarray (N, 3)
    ratio : float
        Target ratio of faces to keep (0 to 1).

    Returns
    -------
    faces_reduced : ndarray
    vertices : ndarray
    """
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(vertices, faces)
        ms.add_mesh(m)
        target = int(faces.shape[0] * ratio)
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target)
        reduced = ms.current_mesh()
        return reduced.face_matrix(), reduced.vertex_matrix()
    except ImportError:
        pass

    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        target = int(faces.shape[0] * ratio)
        mesh_simplified = mesh.simplify_quadric_decimation(target)
        return (np.asarray(mesh_simplified.triangles),
                np.asarray(mesh_simplified.vertices))
    except ImportError:
        pass

    # Fallback: keep a random subset of faces
    n_keep = max(1, int(faces.shape[0] * ratio))
    idx = np.random.choice(faces.shape[0], n_keep, replace=False)
    return faces[idx], vertices


def save_stl(filename, faces, vertices):
    """Save mesh as binary STL."""
    from struct import pack

    normals = np.zeros((faces.shape[0], 3))
    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm > 0:
            n /= norm
        normals[i] = n

    with open(filename, 'wb') as f:
        f.write(b'\0' * 80)  # header
        f.write(pack('<I', faces.shape[0]))
        for i in range(faces.shape[0]):
            f.write(pack('<3f', *normals[i]))
            for j in range(3):
                f.write(pack('<3f', *vertices[faces[i, j]]))
            f.write(pack('<H', 0))


def main():
    parser = argparse.ArgumentParser(description='Marching-Primitives Demo')
    parser.add_argument('csv_file', help='Path to the SDF .csv file')
    parser.add_argument('--ply', help='Path to watertight .ply file for comparison',
                        default=None)
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save output files')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display plots')
    args = parser.parse_args()

    # Load SDF from CSV
    print(f"Loading SDF from {args.csv_file}...")
    raw = np.loadtxt(args.csv_file, delimiter=',').flatten()

    # Parse grid metadata from first 7 values
    grid_size = int(raw[0])
    voxel_grid = {
        'size': np.array([grid_size, grid_size, grid_size]),
        'range': raw[1:7],
    }
    sdf = raw[7:]

    voxel_grid['x'] = np.linspace(voxel_grid['range'][0], voxel_grid['range'][1], grid_size)
    voxel_grid['y'] = np.linspace(voxel_grid['range'][2], voxel_grid['range'][3], grid_size)
    voxel_grid['z'] = np.linspace(voxel_grid['range'][4], voxel_grid['range'][5], grid_size)

    # Build 3D point grid (using 'ij' indexing like MATLAB's ndgrid)
    gx, gy, gz = np.meshgrid(voxel_grid['x'], voxel_grid['y'], voxel_grid['z'], indexing='ij')
    voxel_grid['points'] = np.column_stack([gx.ravel(order='F'), gy.ravel(order='F'), gz.ravel(order='F')]).T  # (3, N)

    voxel_grid['interval'] = ((voxel_grid['range'][1] - voxel_grid['range'][0]) /
                               (grid_size - 1))
    voxel_grid['truncation'] = 1.2 * voxel_grid['interval']
    voxel_grid['disp_range'] = [-np.inf, voxel_grid['truncation']]
    visualize_arclength = 0.01 * np.sqrt(voxel_grid['range'][1] - voxel_grid['range'][0])

    # Truncate SDF
    sdf = np.clip(sdf, -voxel_grid['truncation'], voxel_grid['truncation'])

    # Run Marching-Primitives
    print("Running Marching-Primitives...")
    t_start = time.time()
    x = MPS(sdf, voxel_grid)
    elapsed = time.time() - t_start
    print(f"MPS completed in {elapsed:.2f} seconds. Found {x.shape[0]} superquadrics.")

    # Triangulation and compression
    print("Generating mesh from superquadrics...")
    mesh_original = mesh_superquadrics(x, arclength=visualize_arclength)

    # Mesh compression
    mesh_faces, mesh_vertices = reduce_mesh(
        mesh_original['f'], mesh_original['v'], ratio=0.1)

    # Save results
    pathname = os.path.dirname(args.csv_file)
    name = os.path.splitext(os.path.basename(args.csv_file))[0]

    if not args.no_save:
        # Save superquadric parameters
        sq_path = os.path.join(pathname, f"{name}_sq.npz")
        np.savez(sq_path, x=x.astype(np.float32))
        print(f"Saved superquadric parameters to {sq_path}")

        # Save STL
        stl_path = os.path.join(pathname, f"{name}_sq.stl")
        save_stl(stl_path, mesh_faces, mesh_vertices)
        print(f"Saved mesh to {stl_path}")

        # Save all superquadric parameters as a single CSV (one row per primitive)
        sq_csv_path = os.path.join(pathname, f"{name}_sq.csv")
        header = 'eps1,eps2,ax,ay,az,eul_z,eul_y,eul_x,tx,ty,tz'
        np.savetxt(sq_csv_path, x, delimiter=',', header=header, comments='')
        print(f"Saved {x.shape[0]} superquadric parameters to {sq_csv_path}")

    if args.no_display:
        return

    # Visualization
    view_vector = (151, -40)
    color = np.array([145, 163, 176]) / 255.0
    grid_range = voxel_grid['range']

    def setup_axes(ax, title_str):
        ax.set_xlim(grid_range[0], grid_range[1])
        ax.set_ylim(grid_range[2], grid_range[3])
        ax.set_zlim(grid_range[4], grid_range[5])
        ax.set_aspect('equal')
        ax.view_init(elev=view_vector[1], azim=view_vector[0])
        ax.set_axis_off()
        ax.set_title(title_str)

    # Figure 2: Superquadric mesh with different colors per primitive
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    cmap = plt.cm.get_cmap('tab20', x.shape[0])
    for i in range(x.shape[0]):
        x_mesh, y_mesh, z_mesh = single_mesh_superquadrics(
            x[i], arclength=visualize_arclength, taper=False
        )
        f_i, v_i = mesh2tri(x_mesh, y_mesh, z_mesh, 'f')
        s = f_i.shape[0]
        idx1 = np.arange(s // 8, 3 * s // 8)
        idx2 = np.arange(5 * s // 8, 7 * s // 8)
        f_i = f_i[np.concatenate([idx1, idx2])]
        ax2.plot_trisurf(v_i[:, 0], v_i[:, 1], v_i[:, 2],
                         triangles=f_i, color=cmap(i), edgecolor='none', alpha=1.0)
    setup_axes(ax2, 'Superquadrics from Marching Primitives')

    # Figure 1: Ground truth (if PLY provided)
    if args.ply:
        ply_path = args.ply
        if not os.path.exists(ply_path):
            # Try auto-detecting
            auto_ply = os.path.join(pathname, f"{name}_watertight.ply")
            if os.path.exists(auto_ply):
                ply_path = auto_ply

        if os.path.exists(ply_path):
            print(f"Loading ground truth mesh from {ply_path}...")
            tri_gt, pts_gt = plyread(ply_path, 'tri')

            fig1 = plt.figure(figsize=(8, 6))
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.plot_trisurf(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2],
                             triangles=tri_gt, color=color, edgecolor='none', alpha=1.0)
            setup_axes(ax1, 'Ground truth mesh from marching cubes')

            # Figure 3: Overlap
            fig3 = plt.figure(figsize=(8, 6))
            ax3 = fig3.add_subplot(111, projection='3d')
            ax3.plot_trisurf(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2],
                             triangles=tri_gt, color='g', edgecolor='none', alpha=0.5)
            ax3.plot_trisurf(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2],
                             triangles=mesh_faces, color=color, edgecolor='none', alpha=1.0)
            setup_axes(ax3, 'Overlapping recovered representation with ground truth')

    plt.show()


if __name__ == '__main__':
    main()
