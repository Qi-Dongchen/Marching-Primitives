import numpy as np
import os
import argparse
import csv
import trimesh
import mesh2sdf
import sys
from pathlib import Path


def convert_glb_to_sdf(glb_path, grid_resolution=100, level=2.0, normalize=False):
    """Convert a single GLB file to SDF (CSV) and watertight mesh."""
    glb_path = Path(glb_path)
    assert glb_path.is_file(), f"Input file does not exist: {glb_path}"

    dir_name = glb_path.parent
    stem = glb_path.stem

    output_dir = dir_name / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_normalized" if normalize else ""
    ply_path = output_dir / f"{stem}{suffix}_watertight.ply"
    stl_path = output_dir / f"{stem}{suffix}_watertight.stl"
    csv_path = output_dir / f"{stem}{suffix}.csv"

    # Load GLB — trimesh handles GLB/GLTF natively
    scene = trimesh.load(str(glb_path))
    if isinstance(scene, trimesh.Scene):
        # Merge all meshes in the scene into one
        mesh = scene.dump(concatenate=True)
    else:
        mesh = scene
    print(f"Mesh loaded from {glb_path} (vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)})")

    mesh_scale = 0.8
    size = grid_resolution
    lev = level / size

    # Normalize mesh to [-mesh_scale, mesh_scale]
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    # Compute watertight mesh and SDF
    print("Converting to watertight mesh...")
    sdf, watertight = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=lev, return_mesh=True
    )

    resolution = grid_resolution

    if normalize:
        watertight.export(str(ply_path))
        watertight.export(str(stl_path))
        print(f"Watertight mesh saved to {ply_path}")

        grid_config = np.array(
            [[resolution],
             [-1], [1],
             [-1], [1],
             [-1], [1]]
        )
        writevoxel = np.reshape(np.swapaxes(sdf, 0, 2), (resolution**3, 1))
        writevoxel = np.append(grid_config, writevoxel).reshape(-1, 1)
    else:
        watertight.vertices = watertight.vertices / scale + center
        watertight.export(str(ply_path))
        watertight.export(str(stl_path))
        print(f"Watertight mesh saved to {ply_path}")

        grid_config = np.array(
            [[resolution],
             [-1 / scale + center[0]], [1 / scale + center[0]],
             [-1 / scale + center[1]], [1 / scale + center[1]],
             [-1 / scale + center[2]], [1 / scale + center[2]]]
        )
        writevoxel = np.reshape(np.swapaxes(sdf / scale, 0, 2), (resolution**3, 1))
        writevoxel = np.append(grid_config, writevoxel).reshape(-1, 1)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerows(writevoxel)
    print(f"SDF saved to {csv_path}")

    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert GLB files to Signed Distance Fields (SDF). "
                    "Supports single file or batch processing."
    )
    parser.add_argument(
        "input",
        help="Path to a single .glb file or a directory containing .glb files for batch processing.",
    )
    parser.add_argument(
        "--grid_resolution", type=int, default=100,
        help="Voxel grid resolution, e.g. 100 means 100^3 (default: 100).",
    )
    parser.add_argument(
        "--level", type=float, default=2.0,
        help="Watertight thicken level (default: 2.0).",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Output normalized mesh and SDF in [-1, 1].",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="When input is a directory, search for .glb files recursively.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()

    # Single file mode
    if input_path.is_file():
        if input_path.suffix.lower() != ".glb":
            print(f"Warning: {input_path} does not have .glb extension, attempting anyway.")
        convert_glb_to_sdf(input_path, args.grid_resolution, args.level, args.normalize)
        return 0

    # Batch mode
    if input_path.is_dir():
        pattern = "**/*.glb" if args.recursive else "*.glb"
        glb_files = sorted(input_path.glob(pattern))
        if not glb_files:
            print(f"No .glb files found in {input_path}")
            return 0

        print(f"Found {len(glb_files)} GLB file(s) to process.\n")
        failed = []
        for i, glb_file in enumerate(glb_files, 1):
            print(f"[{i}/{len(glb_files)}] Processing: {glb_file}")
            try:
                convert_glb_to_sdf(glb_file, args.grid_resolution, args.level, args.normalize)
            except Exception as e:
                print(f"  ERROR: {e}")
                failed.append((glb_file, str(e)))
            print()

        if failed:
            print("Failed files:")
            for f, err in failed:
                print(f"  - {f}: {err}")
            return 1

        print(f"Done. Successfully processed {len(glb_files)} GLB file(s).")
        return 0

    print(f"Error: {args.input} is neither a file nor a directory.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
