# DMD Tool (v9.5.0)

A high-performance, GPU-accelerated framework for time-series forecasting and structural analysis using Dynamic Mode Decomposition (DMD).

This application provides a robust suite of tools for sweeping DMD hyperparameters, detecting dominant temporal periods, running smart ensemble forecasts, and managing large-scale matrix decompositions. You provide it with multi-channel time-series data (like sensor readings or financial metrics), and it rapidly sweeps through thousands of combinations of "Window Sizes" (how much history to look at) and "Stack Sizes" (Hankle Time-Delay Embedding depths).

The forecasting pipeline uses one GPU to calculate the mathematical dynamics of the system, predicts the next state, compares it to the actual data to calculate error metrics, and outputs the results in flat tabular files (Parquet/Excel) alongside highly granular, AI-agent-ready HDF5 mathematical tensors.

---

## Key Features

* **Cluster-DMD Forecasting:** A "smart mode" workflow that automatically scans historical data for optimal reconstruction windows based on dominant period detection, generating ensemble forecasts with uncertainty metrics.
* **Independent Meta-Analysis Suite:** Implements a strict separation between "Judgment" and "Math". Specialized, decoupled modules handle **Qualitative Pattern Discovery** (Needle and Gap detection), **Quantitative Statistical Filtering** (Error Thresholding), and **Ensemble Aggregation**, ensuring the core mathematical engine remains focused solely on linear algebra stages.
* **Triple-Sweep Period Discovery:** Restores high-dimensional diagnostic capabilities to identify the "Natural Pulse" of a system. The tool analyzes how accuracy needles distribute and correlate across three distinct dimensions: **Data Row Range** (temporal stability), **Stack Size** (model complexity), and **Window Size** (observation scale).
* **Dominant Period Gap Detection:** A pre-processing signal analyzer that detects dominant temporal frequencies using FFT and Autocorrelation. This helps users narrow down optimal Window Sizes by identifying the natural cyclical "pulse" of the dataset.
* **Training Reconstruction Error Tracking:** Enables the calculation of reconstruction accuracy on the historical training dataset itself, providing deeper insights into how well the DMD operator models the known system dynamics.
* **Micro-Stage Performance Profiling:** Deep hardware benchmarking tools track matrix execution times (Hankel construction, SVD, Operator reduction, and Predictions) down to the millisecond, aggregating results per-row or per-batch for cross-system analysis.
* **Optimized GPU-Accelerated Math Engine with Interchangeable Solvers:** Leverages PyTorch (`torch.linalg.svd`) to process massive time-delay embedded Hankel matrices. Features both a Sequential Mode and a massive 3D Batched Tensor Mode. Users can also dynamically swap core linear algebra solvers (e.g., standard Pseudo-Inverse vs. direct Least-Squares) to maximize GPU stability.
* **Graceful Pause & Save (Ctrl+C):** A single `^C` safely pauses the sweep, finishes the active GPU tensor operation, flushes all data to disk, and updates the recovery file so you can resume execution later.
* **Deadlock-Safe Termination:** Implements a robust two-stage signal interception mechanism allowing graceful shutdowns during heavy C-extension/GPU workloads, preventing data corruption even if the Python GIL is locked by an unresponsive thread.
* **Stateful Resumption & Crash Recovery (`--resume`):** Automatically tracks your hyperparameter configuration using a lightweight state file. If a sweep is interrupted by a power loss or system crash, the `--resume` flag seamlessly restarts the process from the exact row and stack size without losing computation time.
* **Intermediate State, Ultra-Granular 1-to-1 HDF5 Storage (Schema v2.0.0):** Writes exact DMD operators, spatial modes, and temporal dynamics to isolated directories. Designed to be perfectly self-describing and natively structured for downstream AI agents.
* **Fault-Tolerant HDF5 Storage:** Matrix decompositions are written to disk atomically at the end of each pipeline stage. This ensures a system crash will never corrupt previously generated historical data.
* **Modular Architecture:** Built on the **Cement framework**, ensuring clean separation of concerns between CLI routing, Pandas I/O, HDF5 management, and pure mathematical operations.

---

## System Architecture

Built on the **Cement CLI Framework** and **PyTorch**, the codebase utilizes a highly decoupled, domain-driven structure dividing execution, judgment, and infrastructure:
```text
dynamics-analysis/ 
├── .gitignore              # Git ignore rules for cached files and temp data
├── README.md               # Project documentation, architecture, and usage guide
├── main.py                 # Application bootstrap and global signal handlers
├── analysis/               # Core Mathematical Execution Engines
│   ├── dmd/                # Standard DMD Sweeps and Batched Tensor Workflows
│   └── cluster/            # Smart Mode Optimization and Ensemble logic
├── meta_analysis/          # Judgment, Evaluation, and Pattern Recognition
│   ├── evaluator/          # Base quantitative accuracy metrics and thresholding
│   ├── ensemble/           # Statistical aggregation and uncertainty bounds
│   └── period/             # Structural gap analysis and Triple-Sweep discovery
└── utils/                  # Infrastructure, I/O, and Persistence
    ├── hdf5/               # 1-to-1 Ultra-Granular HDF5 schema generation/repair
    ├── io/                 # Pandas data loading, buffering, and state-resume
    └── reporter/           # Reporting and logging utilities
```
---

## Installation

**Prerequisites**
* Python 3.8+
* CUDA Toolkit (Optional, but highly recommended for PyTorch GPU acceleration)

**Dependencies**
Install the required packages via pip:

pip install cement torch numpy pandas h5py openpyxl pyarrow

*(Note: `openpyxl` is required for reading/writing .xlsx files; `pyarrow` is required for .parquet files).*

---

## Usage Guide

The application utilizes a command-line interface with nested sub-commands managed by the Cement framework.

### 1. DMD Profiler (`dmd`) (Main Sweep & Forecasting)

**Description:** The core mathematical engine of the application. It embeds 1D time-series data into high-dimensional Hankel matrices, performs SVD and DMD, and generates forecasted states. It handles both iterative hyperparameter sweeps and massive parallel 3D tensor batching.

**Key Arguments:**
* `-i, --input`: Source data file (.xlsx or .parquet).
* `--channels`: Channels to analyze (Default: S1 S2 S3 S4 S5).
* `--hdf5 all`: Save full Eigendecomposition and DMD Operators to the output HDF5 file.
* `--svd-gpu`: Force SVD calculations to run on the GPU if available.
* `--perf [con]`: Enable performance profiling timers. Adding `con` streams metrics to the terminal.
* `--dmd-lstsq`: Use a direct Least-Squares solver instead of the default Pseudo-Inverse for DMD modes.
* `--hdf5-dir`: Specify a custom directory path for all HDF5 output.
* `--train-rec`: Calculate the reconstruction error of the training data itself.
* `--schema`: Print the complete output JSON schema definition to the terminal and exit.

**Valid Parameter Ranges:**
* `--start-row`: >= 1 (Default: 1).
* `--end-row`: >= `--start-row` (Default: Final row of dataset).
* `--min-stack`: >= 2 (Default: 5).
* `--max-stack`: > `--min-stack` (Default: 41).
* `--min-window`: >= `--max-stack` (Default: 20).
* `--max-window`: > `--min-window` (Default: 151).
* `--hdf5`: Array strings `['hankle', 'svd', 'dmd-op', 'eigen', 'dmd-modes', 'dmd_amp', 'pred', 'all']` (Default: `[]`).

**Constraints (CRITICAL):**
* **The Stack Boundary Constraint (Pre-Flight Validation):** The distance between your Start Row and End Row (`end_row - start_row`) **must** be equal to or greater than the `--min-stack` size.
* **Batch Mode Historical Constraint:** When using `--batch-mode`, `--max-window` dictates the fixed window size. To predict the target `--start-row`, there must be at least `max-window` historical rows available *before* the start row.
* **Solver Interchangeability:** By default, the math engine uses `torch.linalg.pinv` (Pseudo-Inverse). Passing `--dmd-lstsq` overrides this with a direct Least-Squares solver.
* **Dynamic Max Stack Scaling:** If `--inc-start` is provided without an explicit `--max-stack`, the engine will automatically cap the maximum stack size at half the distance of the active row span.

**Execution Workflows:** **A. The Sequential Engine (Default)** Used when sweeping both Window Sizes and Stack Sizes. The engine steps through the dataset row-by-row, evaluating each requested Stack Size sequentially.
1. **Hankel Matrix Construction:** Embeds the 1D time-series data into a high-dimensional 2D Time-Delay Embedding (Hankel matrix) based on the current observation window size ($w$) and stack depth ($s$).
2. **Truncated SVD:** Performs a Singular Value Decomposition on the Hankel matrix, retaining a fixed dimension ratio ($r = max\_rank * 0.99$) to filter out trailing modes.
3. **Operator & Modes Calculation:** Derives the reduced-order dynamic operator ($\tilde{A}$), its eigenvalues, and the continuous spatial modes ($\Phi$).
4. **Prediction & Reconstruction:** Computes the initial mode amplitudes ($b$) using either a Pseudo-Inverse or exact Least-Squares solver, and projects the system forward in time to reconstruct the forecasted future state.

**B. The 3D Batched Tensor Engine (`--batch-mode`)** Bypasses the sequential row loop for extreme high-speed evaluation on fixed window sizes.
1. **Batched 3D Hankel Unfolding:** Uses PyTorch's `unfold` method to slice the *entire* historical dataset simultaneously. It constructs a massive 3D tensor where the first dimension ($B$) represents thousands of overlapping, sliding temporal windows.
2. **Batched SVD:** Feeds the massive 3D tensor directly into PyTorch's `torch.linalg.svd`. This computes the Truncated SVD for all $B$ historical windows simultaneously in a single, massively parallel GPU operation.
3. **Batched Prediction:** Executes high-speed batched matrix multiplications (`bmm`) to solve for all reduced operators, mode amplitudes, and future predictions across the entire dataset concurrently.
4. **HDF5 Consolidation & Formatting:** Writes the entire 3D tensor batch to a single consolidated HDF5 file per stack size to reduce I/O bottlenecks, then detaches the prediction tensor back to the CPU for tabular formatting.

**C. Sequential vs. Batched: Mathematical Parity and Differences**
While the two engines process data fundamentally differently at the hardware level, they are designed to produce identical mathematical dynamics.
* **Hankel Construction Differences:** The Sequential engine creates independent 2D matrices iteratively, isolating memory allocation per window. The Batched engine utilizes a double `unfold` operation to create a massive 3D tensor that represents the entire sweep space. This requires forcing contiguous memory blocks (`contiguous()`), which significantly increases peak VRAM usage to avoid `cuSOLVER` copy-overhead. However, the exact data points within the respective $X$ and $Y$ matrices remain identical.
* **SVD Calculation Differences:** The Sequential engine computes Singular Values one matrix at a time. The Batched engine hands the 3D tensor over to parallelized `cuBLAS`/`cuSOLVER` routines. Importantly, the rank truncation threshold (`r = max(1, int(max_rank * 0.99))`) in both algorithms is based purely on the matrix coordinate dimensions (which are static for a given $w$ and $s$), rather than dynamically calculated cumulative energy. As a result, the exact same number of modes is retained in both engines.
* **Effect on Predictions:** Because the Hankel input matrices and the SVD truncation dimensions are structurally identical, the predictions generated by the Sequential and Batched workflows are mathematically equivalent. The only observable variance in the final forecasted values (typically at $10^{-6}$ precision or smaller) stems entirely from floating-point non-determinism—which naturally occurs during the parallel summation reductions of the batched GPU solvers—and not from algorithmic divergence.

**Example Commands:**
```bash
# Basic Forwards Sweep (Increasing Start Row)
python main.py dmd --input data.xlsx --inc-start --start-row 1 --end-row 500 --min-stack 5 --max-stack 20

# High-Performance Batched Tensor Workflow (Fixed Window)
python main.py dmd --input data.parquet --batch-mode --max-window 150 --start-row 200 --end-row 5000

# Basic Backwards Sweep (Decreasing End Row)
python main.py dmd --input data.parquet --dec-end --min-window 20 --max-window 150 --hdf5 all

# Resume an Interrupted Run
python main.py dmd --input data.parquet --resume
```

---

### 2. Terminal Feedback & Progress Tracking

Because mathematical sweeps can take hours, the application provides real-time terminal feedback.

**Standard Output (Default):**
* **Sequential Mode:** `Processed Start Row 150 (2.45s)`
* **Batched Mode:** `Processed Stack 5 (Rows: 100-250, Window: 150) (14.22s)`

**Benchmarking & Performance Profiling (`--perf`):**
* Appending `--perf` silently tracks micro-stage execution times and generates a detailed diagnostic file.
* Appending `--perf con` overrides standard indicators and streams granular, stage-by-stage timings directly to the console.

---

### 3. Cluster-DMD Forecasting (`cluster`) (Smart Mode)

**Description:**
The Cluster-DMD workflow automates hyperparameter selection. It first runs an Optimization Phase on recent historical data to find the exact Window and Stack configurations that drop the prediction error below your target threshold. Once the optimum "needle" configuration is found, it enters the Ensemble Phase to generate a final, highly resilient prediction alongside uncertainty/variance metrics.

**Workflow:**
1. **Target Resolution:** Determines if you are forecasting a historical row (backtesting) or a true future state (Row N+1).
2. **Search Space Construction:** If provided a `--sweep-input`, optimizes the search space to scan window sizes that are multiples of the resonant period.
3. **Optimization Phase:** Identifies the absolute best Window and Stack combination. Supports both Sequential and Batched modes.
4. **Ensemble Forecast Phase:** Computes multiple predictions using a cluster of surrounding window sizes (`w_opt +/- ensemble_width`).
5. **Statistical Aggregation:** Aggregates predictions to calculate Median Prediction and Standard Deviation (uncertainty).

**Command Line Options:**
* `-i, --input` *(Required)*: Path to the source data file.
* `-o, --output` *(Optional)*: Base name prefix for output files.
* `--error-threshold` *(Optional)*: Target error percentage required to accept an optimal window (Default: 1.0).
* `--forecast-row` *(Optional)*: Specific row index to forecast.
* `--ensemble-width` *(Optional)*: +/- variance in window size used when clustering (Default: 5).
* `--sweep-input` *(Optional)*: Path to a previous sweep file for Period Detection optimization.
* `--batch-mode` *(Optional)*: Enables highly-optimized Batched Tensor Workflow for Optimization.

**Example Commands:**
```bash
# Basic cluster forecast for the future (Row N+1)
python main.py cluster -i data.parquet --error-threshold 2.5 --ensemble-width 10

# Smart backtesting using a previously detected period
python main.py cluster -i data.xlsx --forecast-row 500 --sweep-input previous_sweep.parquet
```

---

### 4. Analysis Tools (`period`) (Period Detection)

**Description:** A post-processing diagnostic tool that detects dominant temporal frequencies by analyzing the output of a previous hyperparameter sweep. It isolates local minima ("needles"), calculates gap distribution, and uses structural correlations to pinpoint optimal Window Sizes.

**Command Line Options:**
* `--start-row` / `--end-row`: Position constraints for the data row.
* `--min-stack` / `--max-stack`: Bounding constraints for Stack sizes.
* `--sw-row` / `--sw-stk` / `--sw-win`: Flags indicating which dimension to actively sweep.
* `--min-err-pct` / `--max-err-pct`: Thresholds to filter accuracy needles.

**Example Command:**
```bash
python main.py period --sw-win --min-err-pct 0.0 --max-err-pct 5.0 --channels S1 S2
```

---

## Data Input

**Requirements:**
* **Excel (.xlsx):** Read with no headers. Columns are mapped sequentially to requested channels.
* **Parquet (.parquet):** Fast, compressed columnar storage. Preferred for large datasets.

**Input Data Schema & Parameter Mapping:**
The core mathematical engine treats your input file as a dense, 2D numerical matrix where **Rows = Time** and **Columns = Variables**.
1. **Rows (Temporal Snapshots):** Every row must represent a single, sequential snapshot in time.
2. **Columns (State Variables):** Mapped positionally from left to right via the `--channels` argument (e.g., `S1`, `S2`).

---

## Data Outputs

### 1. Temporary CSV Buffering & The `--keep-temp` Flag
To prevent memory exhaustion, the IO Manager continuously flushes forecast metrics to `<output_prefix>_temp.csv`. This file acts as the vital recovery ledger. Upon successful completion, it compiles this into your final format and deletes the temp file unless `--keep-temp` is passed.

### 2. The Tabular Record Schema
Each row represents a single prediction generated by a specific Window/Stack combination:
* `data_set_start` / `data_set_end`: Historical slice used.
* `window_size` / `stack_size`: Lookback window and Time-Delay Embedding depth.
* `{channel}_val_target`: Actual ground-truth value.
* `{channel}_pred_value` / `{channel}_pred_err` / `{channel}_err_pct`: Continuous error metrics.
* `{channel}_pred_value_int`: Integer-rounded variants for categorical modeling.

### 3. Performance Output (`_perf.parquet`)
If `--perf` is enabled, records exact sub-millisecond execution times:
* `stage_1_hankel_s`: Hankel Construction time.
* `stage_2_svd_s`: SVD Truncation time.
* `stage_3_operator_s`: Reduced-Order Operator time.
* `stage_4_modes_s`: Spatial DMD Modes time.
* `stage_5_predict_s`: Prediction Reconstruction time.
* `stage_6_hdf5_s`: HDF5 I/O latency.
* `stage_7_format_s`: Tabular formatting time.

### 4. Ultra-Granular HDF5 Output (Schema v2.0.0)
To support downstream Machine Learning workflows, this application utilizes a 1-to-1 HDF5 storage strategy. Every stage of the mathematical pipeline is isolated into its own dedicated directory.

**File Naming Convention:**
Files are named using their exact hyperparameter coordinates:
`_ds_[Data Start]_de_[Data End]_ws_[Window Start]_we_[Window End]_s_[Stack Size].hdf5`

**HDF5 Directory Layout & Matrix Schema:**

<output_prefix>_hdf5/  
├── Hankle/                  # Target: hankle  
│   └── (H, X, Y matrices as float32)
├── SVD_Truncation/          # Target: svd  
│   └── (U, S, Vh, U_r, S_inv, V_r as float32; rank 'r' as int32)
├── Reduced_Operator/        # Target: dmd-op  
│   └── (Atilde as float32)
├── Eigen/                   # Target: eigen  
│   └── (eigvals_real, eigvals_imag, W_eig_real, W_eig_imag as float32)
├── DMD_Modes/               # Target: dmd-modes  
│   └── (Phi_real, Phi_imag as float32)
├── DMD_Amplitudes/          # Target: dmd_amp  
│   └── (b_real, b_imag as float32)
└── Prediction/              # Target: pred
    └── (pred_vec_real as float32)
    
**Schema Element Descriptions:**
* **Hankle**
  * `H`: The full Time-Delay Embedded Hankel Matrix (type: float32).
  * `X`: The current state matrix (Columns 0 to N-1 of H).
  * `Y`: The future state matrix (Columns 1 to N of H).
* **SVD_Truncation**
  * `U` / `S` / `Vh`: The full Left Singular Vectors, Singular Values, and Right Singular Vectors.
  * `U_r` / `S_inv` / `V_r`: The rank-truncated representations used for computation.
  * `r`: The integer value representing the truncation rank boundary.
* **Reduced_Operator**
  * `Atilde`: The reduced-order mathematical operator ($\tilde{A}$) driving the system's dynamics.
* **Eigen**
  * `eigvals_real` / `eigvals_imag`: The real and imaginary components of the continuous system eigenvalues.
  * `W_eig_real` / `W_eig_imag`: The real and imaginary components of the operator's eigenvectors.
* **DMD_Modes**
  * `Phi_real` / `Phi_imag`: The spatial DMD modes ($\Phi$) mapping the dynamics back to the physical space.
* **DMD_Amplitudes**
  * `b_real` / `b_imag`: The initial condition mode amplitudes ($b$).
* **Prediction**
  * `pred_vec_real`: The fully reconstructed, continuous future state forecast vector.

**HDF5 Metadata Attributes:**
Each file is completely self-describing, utilizing a strict metadata schema to ensure traceability.
* **Global Attributes:** Embedded at the root level of the HDF5 file.
  * `data_set_identifier`: A string denoting the specific Data Class generating the file (e.g., "Hankle", "SVD_Truncation").
  * `hierarchical_schema`: A complete JSON string representation of the entire Schema v2.0.0 structure, ensuring downstream agents understand the expected data types without external documentation.
* **Group Attributes:** Attached directly to the specific internal data group (e.g., `Hankle_ds_1_de_500_ws_350_we_500_s_20`).
  * `w_start` / `w_end`: The exact starting and ending row indices of the observation window.
  * `stack_size`: The Time-Delay Embedding depth ($s$) used.
  * `data_class`: The classification of the matrices contained within.
  * `data_types`: A list of the specific arrays contained in the group (e.g., `["H", "X", "Y"]`).
* **Compression Strategy:** To optimize massive SVD sweeps, all multi-dimensional matrices are saved with `gzip` compression (level 4). Singular scalar values (such as the SVD truncation rank `r`) are saved without compression to prevent read overhead.

**HDF5 Metadata Attributes:**
Each file is completely self-describing, utilizing a strict metadata schema to ensure traceability.

* **Global Attributes:** Embedded at the root level of the HDF5 file.
  * `data_set_identifier`: A string denoting the specific Data Class generating the file (e.g., "Hankle", "SVD_Truncation").
  * `hierarchical_schema`: A complete JSON string representation of the entire Schema v2.0.0 structure, ensuring downstream agents understand the expected data types without external documentation.
* **Group Attributes:** Attached directly to the specific internal data group (e.g., `Hankle_ds_1_de_500_ws_350_we_500_s_20`).
  * `w_start` / `w_end`: The exact starting and ending row indices of the observation window.
  * `stack_size`: The Time-Delay Embedding depth ($s$) used.
  * `data_class`: The classification of the matrices contained within.
  * `data_types`: A list of the specific arrays contained in the group (e.g., `["H", "X", "Y"]`).
* **Compression Strategy:** To optimize massive SVD sweeps, all multi-dimensional matrices are saved with `gzip` compression (level 4). Singular scalar values (such as the SVD truncation rank `r`) are saved without compression to prevent read overhead.
* 
### HDF5 Diagnostics & Repair (`hdf5` sub-command)
```bash
# Print the self-describing hierarchical schema of a file

python main.py hdf5 schema output_svd.hdf5

# Inspect a file for truncated B-Trees or corrupt configurations

python main.py hdf5 inspect output_svd.hdf5

# Safely truncate and repair corrupt entries at the tail of a file

python main.py hdf5 fix output_svd.hdf5
```

---

## Fault Tolerance & Crash Recovery

### The Graceful Pause & Deadlock-Safe Termination (Ctrl+C)
1. **First Ctrl+C (Pause & Save):** Sets an internal flag. The engine finishes its current in-flight tensor operation, flushes CSV buffers to disk, updates the execution state file, and exits cleanly.
2. **Second Ctrl+C (Force Kill):** Forces an immediate `os._exit(1)`. Use only if a PyTorch C-extension or HDF5 thread has deadlocked the Python GIL.

### The `--resume` Workflow
When a sweep is initiated, a temporary state file is generated. Passing the `--resume` flag bypasses standard initialization. It reads the state file, automatically restores your original command-line arguments, safely ignores previously calculated matrices, and picks up exactly where it left off.

**Schema Structure:** A flat JSON dictionary mapping the Cement CLI argument namespaces to their runtime values.

Example:
{
    "input": "data.parquet",
    "output": "sweep_results",
    "channels": ["S1", "S2", "S3"],
    "start_row": 1,
    "end_row": 5000,
    "min_stack": 5,
    "max_stack": 41,
    "min_window": 20,
    "max_window": 151,
    "inc_start": true,
    "batch_mode": false,
    "format": "parquet",
    "resume": false,
    "keep_temp": false,
    "train_rec": false,
    "svd_gpu": true,
    "dmd_lstsq": false,
    "perf": null,
    "hdf5": ["all"],
    "hdf5_dir": null
}

**Schema Element Descriptions:**
* `input`: The file path to the source historical data (`.xlsx` or `.parquet`).
* `output`: The base prefix string used for naming all generated output files and directories.
* `channels`: A list of strings identifying which column labels from the input data are actively being analyzed.
* `start_row` / `end_row`: Integer bounds defining the specific temporal span of the dataset targeted for prediction or analysis.
* `min_stack` / `max_stack`: Integer boundaries defining the minimum and maximum Time-Delay Embedding depths to evaluate.
* `min_window` / `max_window`: Integer boundaries defining the minimum and maximum observation window sizes (historical lookback length) to evaluate.
* `inc_start`: Boolean flag indicating if the sweep should iterate forward in time by incrementing the start row.
* `batch_mode`: Boolean flag enabling the massive 3D Batched Tensor math engine instead of the sequential row-loop.
* `format`: String (`csv`, `parquet`, or `xlsx`) dictating the final tabular output file format.
* `resume`: Boolean flag indicating if the current process was launched using the resumption workflow.
* `keep_temp`: Boolean flag dictating if the application should preserve the `_temp.csv` recovery ledger upon successful completion.
* `train_rec`: Boolean flag enabling the calculation of reconstruction error on the historical training data.
* `svd_gpu`: Boolean flag enforcing that Singular Value Decomposition happens on the GPU (default behavior).
* `dmd_lstsq`: Boolean flag activating the exact direct Least-Squares solver instead of the standard Pseudo-Inverse solver.
* `perf`: String modifier (e.g., `"con"`) or `null` dictating the verbosity and behavior of the hardware performance profiler.
* `hdf5`: A list of strings indicating which specific mathematical matrices should be committed to disk (e.g., `["hankle", "svd", "all"]`).
* `hdf5_dir`: An optional string path allowing the user to redirect the massive HDF5 output directories to a secondary drive.

*Architectural Note:* The active progress coordinates (the exact target row, window size, and stack size where a crash occurred) are intentionally excluded from this JSON. Instead, the `IOManager` dynamically infers the exact resumption coordinate by parsing the last successfully written row in the `<output_prefix>_temp.csv` data buffer. This two-file state tracking design guarantees that the configuration boundaries and the actual data committed to the hard drive are always perfectly synchronized, preventing data duplication or gaps upon resumption.

---

## Extending the Framework

To add new workflows or algorithms, create a new Controller class in the appropriate domain directory (`analysis/`, `meta_analysis/`, or `utils/`) and register it in the `handlers` list within `main.py`. Keep heavy mathematical operations isolated in `_main.py` files to maintain framework independence and decouple logic from CLI routing.
