# Marching-Primitives: Shape Abstraction from Signed Distance Function
Project|Paper|Supplementary|[Arxiv](https://arxiv.org/abs/2303.13190)|[3D-Demos](/examples)|[Data](/MATLAB/data)

<img src="/examples/example.jpg" alt="example" width="600"/>

This repo provides the source code for the CVPR2023 paper:
> [**Marching-Primitives: Shape Abstraction from Signed Distance Function**](https://arxiv.org/abs/2303.13190 "ArXiv version of the paper.")  
> [Weixiao Liu](https://github.com/bmlklwx)<sup>1,2</sup>, Yuwei Wu<sup>2</sup>, [Sipu Ruan](https://ruansp.github.io/)<sup>2</sup>, [Gregory S. Chirikjian](https://cde.nus.edu.sg/me/staff/chirikjian-gregory-s/)<sup>2</sup>  
> <sup>1</sup> National University of Singapore, <sup>2</sup> Johns Hopkins University

## Citation
If you find this repo useful, please give us a star and cite:
```
@Inproceedings{Liu2023CVPR,
     title = {Marching-Primitives: Shape Abstraction from Signed Distance Function},
     author = {Liu, Weixiao and Wu, Yuwei and Ruan, Sipu and Chirikjian, Gregory},
     booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
     year = {2023}
}
```
Thanks for your interest!

## Update
 - **March 27th, 2023** - V0.1 basic version is online, including MATLAB implementation of the algorithm, data(SDF) preprocess script, and visualization tools.
Python implementation is planned in the coming updates.
 - **March 28th, 2023** - implementation details has been updated.
 - **April 2nd, 2023** - update bounding-box based parameter reinitialization for better performance (line 101-111 in MPS.m).

## Abstraction
Representing complex objects with basic geometric primitives has long been a topic in computer vision. Primitive-based representations have the merits of compactness and computational efficiency in higher-level tasks such as physics simulation, collision checking, and robotic manipulation. Unlike previous works which extract polygonal meshes from a signed distance function (SDF), in this paper, we present a novel method, named Marching-Primitives, to obtain a primitive-based abstraction directly from an SDF. Our method grows geometric primitives (such as superquadrics) iteratively by analyzing the connectivity of voxels while marching at different levels of signed distance. For each valid connected volume of interest, we march on the scope of voxels from which a primitive is able to be extracted in a probabilistic sense and simultaneously solve for the parameters of the primitive to capture the underlying local geometry. We evaluate the performance of our method on both synthetic and real-world datasets. The results show that the proposed method outperforms the state-of-the-art in terms of accuracy, and is directly generalizable among different categories and scales.

## Implementation
### Marching-Primitives algorithm
The source code of the algorithm is in `src/marching_primitives`
```
x = MPS(sdf, grid)
```
The algorithm depends on the Image Processing Toolbox of MATLAB.
The algorithm requires a Signed Distance Function discretized on a voxel grid as input. More specifically, for a grid of size $(x,y,z):M\times N\times W$, `grid.size = [M, N, W]` is the size of the voxel grid; `grid.range = [x_min, x_max, y_min, y_max, z_min, z_max]` stores the range of the voxel grid, and `sdf` is a 1-D array flattened from the 3-D array storing the signed distance of points in the voxel grid. 


The output of the function is a 2D array of size $K*11$, where each row stores the parameter of a superquadric $[\epsilon_1, \epsilon_2, a_x, a_y, a_z, euler_z, euler_y, euler_x, t_x, t_y, t_z]$.

### Preparing SDF from meshes
If you do not have SDF files but want to test the algorithm, we provide Python scripts to generate SDF from meshes. The scripts are based on the [mesh2sdf](https://github.com/wang-ps/mesh2sdf) package.
The mesh file will be first transformed to be watertight so that a valid SDF can be extracted.

Install the dependency:
```
pip install mesh2sdf trimesh
```

**Single file** — convert a mesh (OBJ, STL, PLY, etc.) to a watertight mesh and SDF:
```bash
python scripts/mesh2sdf_convert.py path/to/mesh --normalize --grid_resolution 100
```

**Batch** — process all mesh files in a directory:
```bash
python scripts/batch_mesh2sdf_convert.py --data_dir data --grid_resolution 100 --normalize
```

Options: `--normalize` normalizes the mesh and SDF within $[-1, 1]$; `--grid_resolution` specifies the voxel grid resolution (default $100$); `--level` sets the watertight thicken level (default $2.0$, GLB only).
The scripts accept: .stl, .off, .ply, .collada, .json, .dict, .glb, .dict64, .msgpack, .obj.

**Output** (saved in a subfolder named after the input file):
- `<name>_watertight.ply` / `<name>_watertight.stl` — watertight mesh
- `<name>.csv` — SDF as CSV

A few meshes from ShapeNet/ModelNet are prepared in the [data](/data).

# Python Scripts Guide

## Installation

```bash
# Install the marching-primitives package (from repo root)
pip install -e .

# Install additional dependencies for mesh conversion scripts
pip install mesh2sdf trimesh

# Optional: for comparison and advanced visualization
pip install scipy scikit-image open3d
```

After installation, the `marching-primitives` CLI command becomes available:
```bash
marching-primitives data/chair1/chair1_normalized.csv --ply data/chair1/chair1_normalized_watertight.ply
```

## Scripts

### main.py

Run the Marching-Primitives (MPS) algorithm on an SDF to extract superquadric primitives, then visualize the results.

```bash
# Basic usage
python scripts/main.py data/chair1/chair1_normalized.csv

# With ground-truth PLY overlay
python scripts/main.py data/chair1/chair1_normalized.csv --ply data/chair1/chair1_normalized_watertight.ply

# Save outputs without displaying plots
python scripts/main.py data/chair1/chair1_normalized.csv --no-display

# Skip saving output files
python scripts/main.py data/chair1/chair1_normalized.csv --no-save
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_file` | positional | — | Path to the SDF `.csv` file |
| `--ply` | str | None | Path to watertight `.ply` file for ground-truth comparison |
| `--no-save` | flag | off | Do not save output files |
| `--no-display` | flag | off | Do not display plots |

**Output** (saved alongside the input CSV):
- `<name>_sq_py.npz` — superquadric parameters (NumPy format)
- `<name>_sq_py.csv` — superquadric parameters (CSV, 11 columns: eps1, eps2, ax, ay, az, eul_z, eul_y, eul_x, tx, ty, tz)
- `<name>_sq_py.stl` — reconstructed mesh from superquadrics

---

## Typical Workflow

```
1. Convert mesh to SDF:
   mesh2sdf_convert.py  (single OBJ/STL/PLY)
   batch_mesh2sdf_convert.py  (batch OBJ)
   glb2sdf_convert.py  (single or batch GLB)
         |
         v
2. Run Marching-Primitives:
   main.py  (extract superquadrics from SDF)
         |
         v
3. Visualize / Compare:
   visualize_sdf.py  (inspect the SDF)
   compare_sq.py  (compare superquadrics vs ground truth)
   visualize_superquadrics.py  (batch render to PNG)
```

## Difference between MATLAB version and Python version

The Python results may differ from the MATLAB results, shown in a different number of reconstructed superquadrics; however, the overall effect does not differ significantly.