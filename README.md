# DMD Tool (v9.3.32)

A high-performance, GPU-accelerated framework for time-series forecasting and structural analysis using Dynamic Mode Decomposition (DMD).

This application provides a robust suite of tools for sweeping DMD hyperparameters, detecting dominant temporal periods, running smart ensemble forecasts, and managing large-scale matrix decompositions. You provide it with multi-channel time-series data (like sensor readings or financial metrics), and it rapidly sweeps through thousands of combinations of "Window Sizes" (how much history to look at) and "Stack Sizes" (Hankle Time-Delay Embedding depths).

The forecasting pipeline uses one GPU to calculate the mathematical dynamics of the system, predicts the next state, compares it to the actual data to calculate error metrics, and outputs the results in flat tabular files (Parquet/Excel) alongside highly granular, AI-agent-ready HDF5 mathematical tensors.

---

## Key Features

* **GPU-Accelerated Math Engine:** Leverages PyTorch (`torch.linalg.svd`) to process massive time-delay embedded Hankel matrices across multiple channels simultaneously.
* **Modular Architecture:** Built on the Cement framework, ensuring clean separation of concerns between CLI routing, Pandas I/O, HDF5 management, and pure mathematical operations.
* **Cluster-DMD Forecasting:** A "smart mode" workflow that automatically scans historical data for optimal reconstruction windows based on dominant period detection, generating ensemble forecasts with uncertainty metrics.
* **Fault-Tolerant HDF5 Storage:** Writes exact DMD operators, spatial modes, and temporal dynamics to self-describing, hierarchically structured HDF5 files. Includes custom B-tree repair tools to recover data from interrupted runs.
* **Deadlock-Safe Termination:** Implements a two-stage signal interception mechanism allowing graceful shutdowns during heavy C-extension/GPU workloads, preventing data corruption.

---

## System Architecture

Built on the **Cement CLI Framework** and **PyTorch**, The codebase has been decoupled from a monolithic script into a scalable MVC-like structure using Cement:
```
dmd_profiler/ 
├── .gitignore              # Git ignore rules for cached files and temp data
├── README.md               # Project documentation, architecture, and usage guide
├── main.py                 # Application bootstrap and global signal handlers
├── core/                   # Framework-agnostic business logic
│   ├── cluster_engine.py   # K-Means clustering logic for dynamic behaviors
│   ├── hdf5_manager.py     # 1-to-1 Ultra-Granular HDF5 schema generation and injection
│   ├── io_manager.py       # Pandas data loading, buffering, state-resume, and exporting
│   ├── math_engine.py      # PyTorch SVD/DMD tensor math (Sequential & 3D Batched Workflows)
│   ├── period_analysis.py  # Algorithms for dominant period gap detection
│   └── reporter.py         # Reporting and logging utilities
└── controllers/            # Cement CLI Routing and Argument Parsing
    ├── analysis_tools.py   # Sub-commands for standalone period analysis
    ├── base.py             # Default execution (Standard Sweeps & Batched Tensor Workflow)
    ├── cluster_tools.py    # Sub-commands for behavior clustering
    └── hdf5_tools.py       # Sub-commands for HDF5 repair and inspection
```
---

## Installation
**Prerequisites**

* Python 3.8+
* CUDA Toolkit (Optional, but highly recommended for PyTorch GPU acceleration)

**Dependencies**
Install the required packages via pip:

**Bash**
`pip install cement torch numpy pandas h5py openpyxl pyarrow`

*(Note: openpyxl is required for reading/writing .xlsx files; pyarrow is required for .parquet files).*

---

## Usage Guide
The application utilizes a Git-like command-line interface with nested sub-commands.

---
### Subfunction Details
This section details the internal mechanics of each subfunction, the valid bounds for their parameters, and the critical rules governing how those parameters interact (Inter-Parameter Constraints).

#### 1. Base Profiler (Main Sweep & Forecasting)
**Description:** The core mathematical engine of the application. It embeds 1D time-series data into high-dimensional Hankel matrices, performs SVD and DMD, and generates forecasted states. It handles both iterative hyperparameter sweeps and massive parallel 3D tensor batching.

**Key Arguments:**

* `-i, --input`: Source data file (.xlsx or .parquet).
* `--channels`: Channels to analyze (Default: S1 S2 S3 S4 S5).
* `--hdf5 all`: Save full Eigendecomposition and DMD Operators to the output HDF5 file.
* `--svd-gpu`: Force SVD calculations to run on the GPU if available.

**Valid Parameter Ranges:**
* `--start-row`: $\ge 1$ (Default: `1`).
* `--end-row`: $\ge$ `--start-row` (Default: Final row of dataset).
* `--min-stack`: $\ge 2$ (Default: `5`).
* `--max-stack`: $>$ `--min-stack` (Default: `41`).
* `--min-window`: $\ge$ `--max-stack` (Default: `20`).
* `--max-window`: $>$ `--min-window` (Default: `151`).
* `--hdf5`: Array strings `['hankle', 'svd', 'dmd-op', 'eigen', 'dmd-modes', 'dmd_amp', 'pred', 'all']` (Default: `[]`).

**Constraints (CRITICAL):**
* **The Stack Boundary Constraint:** The distance between your Start Row and End Row (`end_row - start_row`) **must** be equal to or greater than the `--min-stack` size. If it is smaller, a Hankel matrix cannot be mathematically formed, and the application will abort with a pre-flight validation warning.
* **Batch Mode Historical Constraint:** When using `--batch-mode`, `--max-window` dictates the fixed window size. To predict the target `--start-row`, there must be at least `max-window` historical rows available *before* the start row in the source file.
* **Dynamic Max Stack Scaling:** If `--inc-start` is provided without an explicit `--max-stack`, the engine will automatically cap the maximum stack size at half the distance of the active row span (`(end_row - start_row) // 2`) to prevent non-sensical tall/skinny matrix embeddings.

**Workflow:**
1. Parses CLI inputs and checks for an interrupted run state (`--resume`).
2. Loads the target data slice from Parquet or Excel.
3. Executes pre-flight validation to ensure matrix mathematical viability.
4. Routes data to either the Sequential Loop or the Batched Tensor Engine.
5. Executes the DMD stages: Embed $\to$ SVD $\to$ Operator $\to$ Modes $\to$ Predict.
6. Emits targeted mathematical stages to isolated HDF5 files.
7. Flushes forecast error metrics to a temporary CSV buffer, merging to Parquet/XLSX upon completion.

**Example Commands:**
```bash
# Basic Forwards Sweep (Increasing Start Row)
python main.py --input data.xlsx --inc-start --start-row 1 --end-row 500 --min-stack 5 --max-stack 20

# High-Performance Batched Tensor Workflow (Fixed Window)
python main.py --input data.parquet --batch-mode --max-window 150 --start-row 200 --end-row 5000

# Resume an Interrupted Run
python main.py --input data.parquet --resume

# Standard Parameter Sweeps (Base Controller)
# The default execution mode runs heavy matrix sweeps across a range of window and stack sizes.

# Basic Forwards Sweep (Increasing Start Row):
```bash
python main.py --input data.xlsx --inc-start --start-row 1 --end-row 500 --min-stack 5 --max-stack 20`

# Basic Backwards Sweep (Decreasing End Row):
python main.py --input data.parquet --dec-end --min-window 20 --max-window 150 --hdf5-all`
```

---
#### 2. Analysis Tools (Period Detection)
**Description:**
A pre-processing signal analyzer that detects dominant temporal frequencies in the dataset. This helps users narrow down optimal Window Sizes for the main DMD Profiler by identifying the natural "pulse" of the data.

**Workflow:**
1. Loads requested data channels.
2. Applies a Fast Fourier Transform (FFT) to convert the time-domain signal to the frequency domain.
3. Computes Autocorrelation to verify repeating cyclical gaps.
4. Filters results through Peak Detection algorithms to output the top resonant frequencies.

**Valid Parameter Ranges:**
* `--min-period`: Integer $\ge 2$ (Default: `2`).
* `--max-period`: Integer $\le$ Total rows in dataset.

**Constraints:**
* **Band-Pass Filtering:** `--max-period` must be strictly greater than `--min-period`. 
* **Nyquist Limit:** A period of `1` cannot be detected. The `--min-period` is hard-limited to 2 to satisfy the Nyquist-Shannon sampling theorem (requiring at least two data points to establish a cycle).

**Example Command:**
```bash
python main.py analysis period sweep_results.parquet --channel S1
```
---

#### 3. Cluster Tools
**Description:**
Segments historical system behaviors into distinct dynamic "regimes" using K-Means clustering. Useful for labeling periods of high volatility versus stability prior to routing them into the DMD engine.

**Workflow:**
1. Extracts rolling features (mean, variance, volatility) from the target channels over a sliding window.
2. Standardizes the extracted features to a mean of 0 and standard deviation of 1.
3. Executes the K-Means algorithm.
4. Appends a new `Cluster_Label` column to the dataset and outputs the file.

**Valid Parameter Ranges:**
* `--n-clusters`: Integer $\ge 2$ (Default: `3`).
* `--rolling-window`: Integer $\ge 2$ (Default: `10`).

**Constraints:**
* **Feature Viability:** The size of the dataset must be substantially larger than the `--rolling-window` and `--n-clusters` combined, otherwise the algorithm will lack sufficient distinct data points to form meaningful centroids.

**Example Command:**
```bash
python main.py cluster --input data.parquet --n-clusters 4
```
---

#### 4. Cluster-DMD Forecasting

Enable the "Smart Mode" to automatically optimize window/stack sizes based on historical error thresholds and generate future predictions.

```bash
`python main.py --input data.xlsx --cluster-dmd --forecast-row 1000 --min-window 50 --max-window 200`
```
---
### Data Input
#### Requirements

The application expects continuous time-series data without headers.

* Excel (.xlsx): Read with no headers. Columns are mapped sequentially to requested channels (e.g., Column 0 is S1, Column 1 is S2).
* Parquet (.parquet): Fast, compressed columnar storage. Preferred for large datasets.

#### 1. Input Data Schema & Parameter Mapping

The core mathematical engine treats your input file as a dense, 2D numerical matrix where **Rows = Time** and **Columns = Variables**. Understanding this schema is critical for accurately configuring your command-line parameters.

##### 1. Rows (Temporal Snapshots)
Every row in your dataset must represent a single, sequential snapshot in time (e.g., $t_0, t_1, t_2$). There should be no missing rows or gaps in the time-series.
* The `--start-row` and `--end-row` CLI parameters act as a vertical data slicer. They dictate the exact contiguous block of time-steps the engine will extract into memory to build the Hankel matrices.
* **Example:** Passing `--start-row 100 --end-row 500` tells the engine to isolate those specific 401 snapshots. If you are using `--batch-mode` with `--max-window 150`, the engine uses rows 100 through 249 to predict row 250, then shifts down one row at a time until it reaches row 500.

##### 2. Columns (State Variables / Channels)
Every column represents a distinct feature, sensor, or mathematical state variable. Because the application explicitly expects **headerless** data files, the columns are mapped positionally from left to right.
* The `--channels` argument is mapped sequentially to the column indices (0-indexed) of your input file.
* **Example:** If your dataset has 5 columns, but you pass `--channels Temp Pressure Velocity`, the engine maps them exactly in order:
  * **Column 0:** Maps to `Temp`
  * **Column 1:** Maps to `Pressure`
  * **Column 2:** Maps to `Velocity`
* Any remaining columns in your dataset (Columns 3 and 4) are safely ignored by the matrix embedding phase, allowing you to feed wide datasets into the profiler while only analyzing specific subsets of variables.

---
### Data Outputs

While the pure mathematical tensors (SVD, Eigenvalues, DMD Modes) are exported to the isolated HDF5 directories, the primary forecasting results and error metrics are exported to flat, tabular files. 

Controlled by the `--format` argument (`parquet`, `xlsx`, or `both`), the application generates a comprehensive summary file containing the exact predictions and error rates for every single row and hyperparameter combination tested during the sweep.

#### 1. The Tabular Record Schema
Each row in the output file represents a single prediction generated by a specific Window/Stack combination. The columns are structured as follows:

**1. Sweep Metadata**
* `data_set_start` / `data_set_end`: The specific historical slice of rows used to build the Hankel matrix for this prediction.
* `window_size`: The temporal lookback window ($w$) used.
* `stack_size`: The Time-Delay Embedding depth ($s$) used.
* `rank_ratio`: The truncation threshold used during SVD (Default: 0.99).

**2. Channel Forecasting Metrics**
For every channel requested (e.g., if you passed `--channels S1 S2`), the output dynamically generates a suite of prediction columns:
* `{channel}_val_target`: The actual ground-truth value from the dataset at the target row.
* `{channel}_pred_value`: The raw, continuous float prediction calculated by the DMD operator.
* `{channel}_pred_err`: The absolute error ($| \text{target} - \text{prediction} |$).
* `{channel}_err_pct`: The percentage error relative to the target value.

*(Note: The engine also generates `_int` appended columns for each metric, which provide rounded integer values for discrete state tracking).*

---
#### 2. Ultra-Granular HDF5 Output (Schema v2.0.0)

To support massive hyperparameter sweeps and downstream Machine Learning workflows, this application utilizes an "Ultra-Granular 1-to-1" HDF5 storage strategy. 

Instead of dumping all data into a single, highly-fragmented monolithic file (which is prone to corruption during power loss), the engine completely isolates every stage of the mathematical pipeline into its own dedicated directory and self-describing file. This ensures that the data is perfectly structured for Agentic AI ingestion and cross-team sharing.

##### Key HDF5 Features
Embedded JSON Schemas: Every single .hdf5 file contains an embedded global attribute called hierarchical_schema. This acts as an internal blueprint, allowing any script (or AI agent) to read the file and instantly understand exactly what datasets, data types, and matrix shapes are inside without having to guess.

**Batch Mode Consolidation**: When using the standard sequential sweep, each time-step and stack size gets an individual atomic file. However, if you enable --batch-mode, the application smartly consolidates the data, outputting exactly ONE file per stack size that contains the entire multi-dimensional 3D batch tensor, preventing file-system bloat while preserving the 1-to-1 isolation.

**Fault-Tolerant**: Because files are written atomically at the end of a mathematical calculation, a system crash will never corrupt the historical data you have already generated.

##### HDF5 Directory Layout

When you run a sweep with the `--hdf5 all` flag (or specific targets), the application generates a root folder based on your `--output` name. Inside, it builds 7 distinct sub-directories mapping directly to the mathematical stages of the Dynamic Mode Decomposition pipeline.

```text
<output_prefix>_hdf5/
├── Hankle/                  # Target: `hankle`
│   └── window_<w>_stack_<s>.hdf5   (Contains: H_batch, X_batch, Y_batch)
├── SVD_Truncation/          # Target: `svd`
│   └── window_<w>_stack_<s>.hdf5   (Contains: U, S, Vh, Truncated r)
├── Reduced_Operator/        # Target: `dmd-op`
│   └── window_<w>_stack_<s>.hdf5   (Contains: Atilde)
├── Eigen/                   # Target: `eigen`
│   └── window_<w>_stack_<s>.hdf5   (Contains: Continuous & Discrete Eigenvalues)
├── DMD_Modes/               # Target: `dmd-modes`
│   └── window_<w>_stack_<s>.hdf5   (Contains: Exact spatial/dynamic modes (Phi))
├── DMD_Amplitudes/          # Target: `dmd_amp`
│   └── window_<w>_stack_<s>.hdf5   (Contains: Mode amplitudes (b))
└── Prediction/              # Target: `pred`
    └── window_<w>_stack_<s>.hdf5   (Contains: The final forecasted real-value vectors)
```
#### 3. HDF5 Diagnostics & Repair (`hdf5`)
**Description:**
A suite of tools for reading and managing the Ultra-Granular HDF5 tensor outputs. Designed primarily to help downstream AI Agents ingest the self-describing schemas.

**Workflow:**
1. Mounts the target `.hdf5` file in read-only mode.
2. Extracts global dictionary attributes.
3. Parses the embedded `hierarchical_schema` JSON.
4. Formats and prints the schema definition to standard output.

**Valid Parameters:**
* `<target_path>`: String representation of a valid file path or directory (Required).

**Inter-Parameter Constraints:**
* If `<target_path>` points to a directory instead of a file, the tool will automatically crawl the directory to find the first `.hdf5` file to extract the shared global schema.

#### HDF5 Diagnostics & Repair (hdf5 sub-command)

Because large SVD sweeps can generate massive HDF5 files, the application includes tools to inspect and repair files corrupted by hard crashes or power loss.

##### Print the self-describing hierarchical schema of a file:

```bash
python main.py hdf5 schema output_svd.hdf5`
```
#### Inspect a file for truncated B-Trees or corrupt configurations:

```bash
python main.py hdf5 inspect output_svd.hdf5`
```
##### Safely truncate and repair corrupt entries at the tail of a file:

```bash
python main.py hdf5 fix output_svd.hdf5`
```
##### Post-Processing Analysis (analysis sub-command)

Run standalone analysis on generated sweep data.

**Detect dominant temporal periods from a previous parameter sweep:**

```bash
python main.py analysis period sweep_results.parquet --channel S1`
```
---
### Graceful Termination

The application implements custom SIGINT and SIGTERM handling to protect your data during long sweeps:

1. First`Ctrl+C`: Sets an internal application flag. The math engine will finish its current tensor operation, write the buffers safely to disk, and exit cleanly.
2. Second`Ctrl+C`: Forces an immediate`os._exit(1)`. Use this only if a PyTorch C-extension or HDF5 thread has completely deadlocked the Python GIL and is unresponsive to the first interrupt.

---

# Extending the Framework

To add new workflows or algorithms, create a new Controller class in the controllers/ directory and register it in the handlers list within main.py. Keep heavy mathematical operations isolated in the core/ directory to maintain framework independence.
