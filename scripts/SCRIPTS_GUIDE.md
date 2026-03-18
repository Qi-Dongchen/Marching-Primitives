# Python Scripts Guide

This folder contains Python scripts for converting meshes to Signed Distance Fields (SDF), running the Marching-Primitives algorithm, and visualizing results.

## Prerequisites

```bash
pip install numpy trimesh matplotlib scikit-image open3d
```

---

## 1. mesh2sdf_convert.py

Convert a single mesh file (OBJ, STL, PLY, etc.) to a watertight mesh and SDF.

```
pip install mesh2sdf
```
Then simply run
```
python mesh2sdf_convert.py path/to/model.obj --normalize --grid_resolution 100
```

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
python scripts/batch_mesh2sdf_convert.py --data_dir data --grid_resolution 100 --normalize
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data_dir` | str | `data` | Directory containing `.obj` files |
| `--grid_resolution` | int | 100 | Voxel grid resolution |
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

## 8. Difference between MATLAB version and Python version

In example folder
|  Name         | #Mat  #Py |  MAT RMSE   PY RMSE |  MAT IoU   PY IoU |  M-P RMSE  |
|  crab_mc      |   49   50 |  0.167250  0.167306 |   0.2474   0.2466 |  0.004092  |
|  crab_mps     |   45   46 |  0.166463  0.166099 |   0.2465   0.2476 |  0.006150  |
|  hind_attack_helicopter_mc |   35   38 |  0.135225  0.135748 |   0.1449   0.1455 |  0.009043  |
|  hind_attack_helicopter_mps |   28   28 |  0.141775  0.139141 |   0.1426   0.1413 |  0.013172  |
|  model_of_a_boat__ncma_explore_mc |   70   67 |  0.222539  0.218818 |   0.1021   0.1023 |  0.034843  |
|  model_of_a_boat__ncma_explore_mps |   71   69 |  0.218630  0.216195 |   0.1016   0.1022 |  0.026821  |
|  model_of_a_boat__ncma_explore_v2_normalized_watertight |   70   67 |  0.222539  0.218818 |   0.1021   0.1023 |  0.034843  |
|  octpus_mc    |   61   66 |  0.107034  0.106641 |   0.3321   0.3339 |  0.007110  |
|  octpus_mps   |   64   61 |  0.110310  0.109912 |   0.3179   0.3170 |  0.006333  |
|  sculpture_mc |   86  135 |  0.228809  0.218657 |   0.1873   0.1911 |  0.051253  |
|  sculpture_mps |   75   73 |  0.229701  0.219506 |   0.1963   0.1977 |  0.052946  |
|  sleek_modern_dining_table_set_mc |   61   63 |  0.107445  0.105558 |   0.2852   0.2859 |  0.017874  |
|  sleek_modern_dining_table_set_mps |   59   64 |  0.107626  0.106886 |   0.2886   0.2886 |  0.020532  |
|  terataner_balrog_mc |   34   36 |  0.140281  0.140605 |   0.1260   0.1217 |  0.003958  |
|  terataner_balrog_mps |   36   36 |  0.141660  0.142006 |   0.1234   0.1234 |  0.004661  |
|  AVERAGE      |           |  0.163152  0.160793 |   0.1963   0.1965 |  0.019575  |

In data folder (chairs)
|  Name         | #Mat  #Py |  MAT RMSE   PY RMSE |  MAT IoU   PY IoU |  M-P RMSE  |
|  chair1       |    6    6 |  0.258723  0.231910 |   0.4503   0.4528 |  0.085715  |
|  chair10      |   27   25 |  0.181301  0.173899 |   0.4879   0.4869 |  0.018053  |
|  chair11      |   98  106 |  0.195202  0.193312 |   0.0932   0.0934 |  0.028430  |
|  chair2       |   22   22 |  0.200452  0.199881 |   0.2903   0.2906 |  0.052458  |
|  chair4       |   42   41 |  0.149312  0.139465 |   0.1307   0.1296 |  0.058784  |
|  chair5       |   21   23 |  0.182158  0.135228 |   0.3493   0.3509 |  0.104290  |
|  chair6       |   14   17 |  0.152956  0.126496 |   0.7160   0.7136 |  0.056170  |
|  chair8       |  118  154 |  0.265801  0.264652 |   0.0525   0.0517 |  0.040609  |
|  chair9       |   20   24 |  0.167114  0.166236 |   0.2226   0.2238 |  0.027678  |
|  AVERAGE      |           |  0.194780  0.181231 |   0.3103   0.3104 |  0.052465  |