# Python Scripts Guide

This folder contains Python scripts for converting meshes to Signed Distance Fields (SDF), running the Marching-Primitives algorithm, and visualizing results.

## Prerequisites

```bash
pip install numpy trimesh mesh2sdf matplotlib scikit-image
# Optional (for mesh decimation): pip install pymeshlab or open3d
# Optional (for STL loading in compare_sq): pip install numpy-stl
```

---

## 1. mesh2sdf_convert.py

Convert a single mesh file (OBJ, STL, PLY, etc.) to a watertight mesh and SDF.

```bash
# Basic usage (outputs original-scale SDF and watertight mesh)
python scripts/mesh2sdf_convert.py path/to/model.obj

# With normalization (SDF in [-1, 1] range)
python scripts/mesh2sdf_convert.py path/to/model.obj --normalize

# Custom grid resolution and thicken level
python scripts/mesh2sdf_convert.py path/to/model.obj --grid_resolution 64 --level 3.0
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `path_to_data` | positional | — | Path to the input mesh file |
| `--grid_resolution` | int | 100 | Voxel grid resolution (e.g., 100 = 100^3) |
| `--level` | float | 2.0 | Watertight thicken level |
| `--normalize` | flag | off | Output normalized mesh and SDF in [-1, 1] |

**Output** (saved in a subfolder named after the input file):
- `<name>_watertight.ply` / `<name>_watertight.stl` — watertight mesh
- `<name>.csv` — SDF as CSV

---

## 2. batch_mesh2sdf_convert.py

Batch process all OBJ files in a directory using `mesh2sdf_convert.py`.

```bash
# Process all .obj files in the data/ directory
python scripts/batch_mesh2sdf_convert.py --data_dir data

# With options
python scripts/batch_mesh2sdf_convert.py --data_dir data --grid_resolution 64 --level 2.0 --normalize
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data_dir` | str | `data` | Directory containing `.obj` files |
| `--grid_resolution` | int | 100 | Voxel grid resolution |
| `--level` | float | 2.0 | Watertight thicken level |
| `--normalize` | flag | off | Enable normalization |

---

## 3. glb2sdf_convert.py

Convert GLB files to SDF. Supports both single file and batch processing (no separate batch script needed).

```bash
# Single file
python scripts/glb2sdf_convert.py path/to/model.glb --normalize

# Batch: all .glb files in a directory
python scripts/glb2sdf_convert.py data/ --normalize

# Batch recursive: search subdirectories too
python scripts/glb2sdf_convert.py data/ --recursive --normalize

# Custom resolution
python scripts/glb2sdf_convert.py data/ --grid_resolution 64 --level 3.0
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `input` | positional | — | Path to a `.glb` file or a directory for batch mode |
| `--grid_resolution` | int | 100 | Voxel grid resolution |
| `--level` | float | 2.0 | Watertight thicken level |
| `--normalize` | flag | off | Output normalized mesh and SDF in [-1, 1] |
| `--recursive` | flag | off | Search subdirectories for `.glb` files |

**Output** (same structure as `mesh2sdf_convert.py`):
- `<name>_watertight.ply` / `<name>_watertight.stl` — watertight mesh
- `<name>.csv` — SDF as CSV

---

## 4. demo_script.py

Run the Marching-Primitives (MPS) algorithm on an SDF to extract superquadric primitives, then visualize the results. This is the Python equivalent of `MATLAB/demo_script.m`.

```bash
# Basic usage
python scripts/demo_script.py data/chair1/chair1_normalized.csv

# With ground-truth PLY overlay
python scripts/demo_script.py data/chair1/chair1_normalized.csv --ply data/chair1/chair1_normalized_watertight.ply

# Save outputs without displaying plots
python scripts/demo_script.py data/chair1/chair1_normalized.csv --no-display

# Skip saving output files
python scripts/demo_script.py data/chair1/chair1_normalized.csv --no-save
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_file` | positional | — | Path to the SDF `.csv` file |
| `--ply` | str | None | Path to watertight `.ply` file for ground-truth comparison |
| `--no-save` | flag | off | Do not save output files |
| `--no-display` | flag | off | Do not display plots |

**Output** (saved alongside the input CSV):
- `<name>_sq.npz` — superquadric parameters (NumPy format)
- `<name>_sq.csv` — superquadric parameters (CSV, 11 columns: eps1, eps2, ax, ay, az, eul_z, eul_y, eul_x, tx, ty, tz)
- `<name>_sq.stl` — reconstructed mesh from superquadrics

---

## 5. visualize_sdf.py

Visualize an SDF from a CSV file using scatter plots, iso-surfaces, or 2D slices.

```bash
# Show all visualization modes (scatter + isosurface + slice)
python scripts/visualize_sdf.py data/chair1/chair1_normalized.csv

# Scatter plot only
python scripts/visualize_sdf.py data/chair1/chair1_normalized.csv --mode scatter

# Iso-surface only (requires scikit-image)
python scripts/visualize_sdf.py data/chair1/chair1_normalized.csv --mode isosurface

# 2D slice along X axis at index 30
python scripts/visualize_sdf.py data/chair1/chair1_normalized.csv --mode slice --slice-axis x --slice-index 30

# Custom threshold and opacity
python scripts/visualize_sdf.py data/chair1/chair1_normalized.csv --threshold 0.01 --alpha 0.5
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_file` | positional | — | Path to the SDF `.csv` file |
| `--mode` | str | `all` | Visualization mode: `scatter`, `isosurface`, `slice`, or `all` |
| `--threshold` | float | 0.0 | SDF threshold (scatter: show \|sdf\| <= t; isosurface: level) |
| `--slice-axis` | str | `z` | Axis for slice mode: `x`, `y`, or `z` |
| `--slice-index` | int | middle | Slice index along the chosen axis |
| `--alpha` | float | 0.3 | Opacity for scatter/isosurface |

---

## 6. compare_sq.py

Compare extracted superquadric primitives against a ground-truth scan (PLY, STL, or SDF CSV).

```bash
# Visualize primitives only
python scripts/compare_sq.py data/chair1/chair1_normalized_sq.csv

# Overlay with ground-truth PLY mesh
python scripts/compare_sq.py data/chair1/chair1_normalized_sq.csv --scan data/chair1/chair1_normalized_watertight.ply

# Overlay with original SDF
python scripts/compare_sq.py data/chair1/chair1_normalized_sq.csv --scan data/chair1/chair1_normalized.csv

# Custom arclength for mesh sampling
python scripts/compare_sq.py data/chair1/chair1_normalized_sq.csv --arclength 0.01
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `sq_csv` | positional | — | Path to superquadric parameters CSV |
| `--scan` | str | None | Path to scan file (PLY, STL, or SDF CSV) for overlay |
| `--arclength` | float | 0.02 | Mesh sampling arclength |

---

## 7. visualize_superquadrics.py

Batch render superquadric primitives from CSV files to PNG images (headless, no display needed).

```bash
# Default: process data/chair*/chair*_normalized_sq.csv
python scripts/visualize_superquadrics.py

# Custom glob pattern
python scripts/visualize_superquadrics.py --pattern "data/**/chair*_sq.csv"

# Custom output directory and DPI
python scripts/visualize_superquadrics.py --output-dir results/images --dpi 300 --arclength 0.01
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--pattern` | str | `data/chair*/chair*_normalized_sq.csv` | Glob pattern for SQ CSV files |
| `--output-dir` | str | `data/sq_visualizations` | Directory to save PNG files |
| `--arclength` | float | 0.02 | Sampling arclength for superquadric meshing |
| `--dpi` | int | 220 | DPI for saved images |

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
   demo_script.py  (extract superquadrics from SDF)
         |
         v
3. Visualize / Compare:
   visualize_sdf.py  (inspect the SDF)
   compare_sq.py  (compare superquadrics vs ground truth)
   visualize_superquadrics.py  (batch render to PNG)
```
