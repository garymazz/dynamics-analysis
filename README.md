# DMD Tool (v9.4.1)

A high-performance, GPU-accelerated framework for time-series forecasting and structural analysis using Dynamic Mode Decomposition (DMD).

This application provides a robust suite of tools for sweeping DMD hyperparameters, detecting dominant temporal periods, running smart ensemble forecasts, and managing large-scale matrix decompositions. You provide it with multi-channel time-series data (like sensor readings or financial metrics), and it rapidly sweeps through thousands of combinations of "Window Sizes" (how much history to look at) and "Stack Sizes" (Hankle Time-Delay Embedding depths).

The forecasting pipeline uses one GPU to calculate the mathematical dynamics of the system, predicts the next state, compares it to the actual data to calculate error metrics, and outputs the results in flat tabular files (Parquet/Excel) alongside highly granular, AI-agent-ready HDF5 mathematical tensors.

---

The forecasting pipeline uses one GPU to calculate the mathematical dynamics of the system, predicts the next state, compares it to the actual data to calculate error metrics, and outputs the results in flat tabular files (Parquet/Excel) alongside highly granular, AI-agent-ready HDF5 mathematical tensors.

**Key Features**

* **Cluster-DMD Forecasting:** A "smart mode" workflow that automatically scans historical data for optimal reconstruction windows based on dominant period detection, generating ensemble forecasts with uncertainty metrics.
* **Dominant Period Gap Detection:** A pre-processing signal analyzer that detects dominant temporal frequencies using FFT and Autocorrelation. This helps users narrow down optimal Window Sizes by identifying the natural cyclical "pulse" of the dataset.
* **Training Reconstruction Error Tracking:** Enables the calculation of reconstruction accuracy on the historical training dataset itself, providing deeper insights into how well the DMD operator models the known system dynamics.
* **Micro-Stage Performance Profiling:** Deep hardware benchmarking tools that track matrix execution times (Hankel construction, SVD, Operator reduction, and Predictions) down to the millisecond, aggregating results per-row or per-batch for cross-system analysis.
* **Optimized GPU-Accelerated Math Engine with Interchangeable Solvers:** Leverages PyTorch (torch.linalg.svd) to process massive time-delay embedded Hankel matrices. Features both a Sequential Mode and a massive 3D Batched Tensor Mode. Users can also dynamically swap core linear algebra solvers (e.g., standard Pseudo-Inverse vs. direct Least-Squares) to maximize GPU stability.
* **Graceful Pause & Save (Ctrl+C):** A single ^C safely pauses the sweep, finishes the active GPU tensor operation, flushes all data to disk, and updates the recovery file so you can resume execution later.
* **Deadlock-Safe Termination:** Implements a robust two-stage signal interception mechanism allowing graceful shutdowns during heavy C-extension/GPU workloads, preventing data corruption even if the Python GIL is locked by an unresponsive thread.
* **Stateful Resumption & Crash Recovery (--resume):** Automatically tracks your hyperparameter configuration using a lightweight state file. If a sweep is interrupted by a power loss or system crash, the \--resume flag seamlessly restarts the process from the exact row and stack size without losing computation time.
* **Intermediate State, Ultra-Granular 1-to-1 HDF5 Storage:** Writes exact DMD operators, spatial modes, and temporal dynamics to isolated directories. Designed to be perfectly self-describing and natively structured for downstream AI agents.
* **Fault-Tolerant HDF5 Storage:** Matrix decompositions are written to disk atomically at the end of each pipeline stage. This ensures a system crash will never corrupt previously generated historical data.
* **Modular Architecture:** Built on the Cement framework, ensuring clean separation of concerns between CLI routing, Pandas I/O, HDF5 management, and pure mathematical operations.

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

```Bash
pip install cement torch numpy pandas h5py openpyxl pyarrow
```

*(Note: openpyxl is required for reading/writing .xlsx files; pyarrow is required for .parquet files).*

---

## Usage Guide
The application utilizes a command-line interface with nested sub-commands.

The application utilizes a command-line interface with nested sub-commands.

### 1. Base Profiler (Main Sweep & Forecasting)

**Description:** The core mathematical engine of the application. It embeds 1D time-series data into high-dimensional Hankel matrices, performs SVD and DMD, and generates forecasted states. It handles both iterative hyperparameter sweeps and massive parallel 3D tensor batching.

**Key Arguments:**

* `-i, --input`: Source data file (.xlsx or .parquet).
* `--channels`: Channels to analyze (Default: S1 S2 S3 S4 S5).
* `--hdf5 all`: Save full Eigendecomposition and DMD Operators to the output HDF5 file.
* `--svd-gpu`: Force SVD calculations to run on the GPU if available.
* `--perf \[con\]`: Enable performance profiling timers. Adding con streams metrics to the terminal.
* `--dmd-lstsq`: Use a direct Least-Squares solver instead of the default Pseudo-Inverse for DMD modes.
* `--hdf5-dir`: Specify a custom directory path for all HDF5 output. If provided alongside `--hdf5`, the isolated stage directories will be generated here instead of the current working directory.
* `--train-rec`: Calculate the reconstruction error of the training data itself, in addition to the predicted future state.
* `--schema`: Print the complete output JSON schema definition to the terminal and exit.

**Valid Parameter Ranges:**

* \--start-row: $\\ge 1$ (Default: 1).
* \--end-row: $\\ge$ \--start-row (Default: Final row of dataset).
* \--min-stack: $\\ge 2$ (Default: 5).
* \--max-stack: $\>$ \--min-stack (Default: 41).
* \--min-window: $\\ge$ \--max-stack (Default: 20).
* \--max-window: $\>$ \--min-window (Default: 151).
* \--hdf5: Array strings \['hankle', 'svd', 'dmd-op', 'eigen', 'dmd-modes', 'dmd\_amp', 'pred', 'all'\] (Default: \[\]).

**Constraints (CRITICAL):**

* **The Stack Boundary Constraint (Pre-Flight Validation):** The distance between your Start Row and End Row (end\_row \- start\_row) **must** be equal to or greater than the \--min-stack size. If the data window is too small to form a matrix, the engine's Pre-Flight Validation Check will intercept the execution, calculate the exact row deficit, and gracefully abort with a formatted terminal warning (preventing Python tracebacks).
* **Batch Mode Historical Constraint:** When using \--batch-mode, \--max-window dictates the fixed window size. To predict the target \--start-row, there must be at least max-window historical rows available *before* the start row in the source file.
* **Solver Interchangeability:** By default, the math engine uses torch.linalg.pinv (Pseudo-Inverse) to find the minimum norm solution for mode amplitudes. Passing \--dmd-lstsq overrides this with a direct Least-Squares solver, which increases mathematical stability for ill-conditioned complex tensors and actively prevents PyTorch from triggering a slow CPU-fallback on GPUs.
* **Dynamic Max Stack Scaling:** If \--inc-start is provided without an explicit \--max-stack, the engine will automatically cap the maximum stack size at half the distance of the active row span to prevent non-sensical tall/skinny matrix embeddings.

**Execution Workflows:** After parsing CLI inputs, loading data, and passing Pre-Flight Validation, the application routes the data into one of two distinct mathematical engines:

**A. The Sequential Engine (Default)** Used when sweeping both Window Sizes and Stack Sizes.

1. **Time-Stepping:** The engine steps through the dataset row-by-row (either forwards or backwards in time).
2. **Stack Iteration:** For a single target row, the engine evaluates every requested Stack Size sequentially.
3. **Execution:** Constructs 2D Hankel matrices, computes the SVD and DMD operator, and generates a prediction for that specific row.
4. **I/O Interleaving:** Saves individual, atomic HDF5 files for each row/stack combination and yields the tabular record.

**B. The 3D Batched Tensor Engine (`--batch-mode`)** Used for massive, high-speed Stack Size sweeps where the Window Size is locked (`--max-window`). It completely bypasses the sequential row loop.

1. **Stack Iteration (Inverted Loop):** The engine iterates through the requested Stack Sizes first.
2. **3D Tensor Unfolding:** For a given stack size, the engine uses PyTorch's`unfold` method to slice the*entire* historical dataset simultaneously. It constructs a massive 3D tensor batch where the first dimension`(B)` represents every sliding temporal window at once.
3. **Parallel GPU Math:** The 3D tensor batch is passed directly into PyTorch's batched linear algebra solvers (`torch.linalg.svd`,`torch.linalg.lstsq`). The SVD truncation, Operator reduction, and Mode calculations are executed across all`B` windows simultaneously in a single, massively parallel GPU step.
4. **HDF5 Consolidation:** Instead of saving thousands of individual files, the engine writes the entire 3D tensor batch to a single, consolidated HDF5 file per stack size, dramatically reducing file-system I/O overhead.
5. **Tabular Formatting:** The massive prediction tensor is detached back to the CPU, unspooled, and formatted into individual row records for the final CSV/Parquet buffer.

**Example Commands:**

```Bash
# Basic Forwards Sweep (Increasing Start Row)
python main.py --input data.xlsx --inc-start --start-row 1 --end-row 500 --min-stack 5 --max-stack 20
# High-Performance Batched Tensor Workflow (Fixed Window)
python main.py --input data.parquet --batch-mode --max-window 150 --start-row 200 --end-row 5000
# Basic Backwards Sweep (Decreasing End Row)
python main.py --input data.parquet --dec-end --min-window 20 --max-window 150 --hdf5 all
# Resume an Interrupted Run
python main.py --input data.parquet --resume
```

---

### 2. Terminal Feedback & Progress Tracking

Because mathematical sweeps can take hours, the application provides real-time terminal feedback to ensure the system is actively computing.

**Standard Output (Default):**

* **Sequential Mode:** The application outputs the specific target row being processed and the total time taken to clear all stack sizes for that row.
  * *Example:* Processed Start Row 150 (2.45s)
* **Batched Mode:** Because batched mode processes all rows simultaneously, the progress indicator outputs whenever a specific*Stack Size* finishes computing across the entire 3D tensor batch.
  * *Example:* Processed Stack 5 (Rows: 100-250, Window: 150\) (14.22s)

**Benchmarking & Performance Profiling (--perf):**

To evaluate the algorithm's speed on different hardware, you can enable deep profiling.

* Appending `--perf` to your command silently tracks micro-stage execution times and generates a detailed diagnostic file.
* Appending `--perf con` overrides the standard progress indicators and streams the highly granular, stage-by-stage timings (Hankel Construction, SVD Truncation, Operator Reduction, etc.) directly to the console under a formatted \[Row Benchmark\] or \[Batch Mode\] header.

---

### 3. Cluster-DMD Forecasting (Smart Mode)

### Description:

The Cluster-DMD workflow is the "Smart Mode" predictive engine. Instead of requiring you to manually guess the best hyperparameters, this tool automates the process. It first runs an Optimization Phase on recent historical data to find the exact Window and Stack configurations that drop the prediction error below your target threshold. Once the optimum "needle" configuration is found, it enters the Ensemble Phase, testing a cluster of window sizes surrounding the optimum to generate a final, highly resilient prediction alongside uncertainty/variance metrics.

**Workflow:**

1. **Target Resolution:** Determines if you are forecasting a known historical row (for backtesting) or a true future state (Row N+1).
2. **Search Space Construction:** If provided with a detected period (via `--sweep-input`), the engine optimizes the search space to only scan window sizes that are multiples of that resonant period. Otherwise, it runs a full spectrum scan.
3. **Optimization Phase:** Identifies the absolute best Window and Stack combination that yields an error below the `--error-threshold`.
   * *Sequential Mode (Default):* The engine steps back one row and iterates through configurations.
   * *Batched Tensor Mode (`--batch-mode`):* The engine extracts a validation block of the last 20 historical rows and evaluates the entire block simultaneously on the GPU, calculating the mean error across the batch for lightning-fast optimization.
4. **Ensemble Forecast Phase:** Always runs sequentially. Using the optimal configuration, the engine computes multiple predictions for the target row using a cluster of surrounding window sizes (`w_opt +/- ensemble_width`).
5. **Statistical Aggregation:** The ensemble of predictions is aggregated to calculate the Median Prediction and Standard Deviation (uncertainty) for each channel.

**Command Line Options:**

The `cluster` sub-command utilizes the following arguments:

* `-i, --input` *(Required)*: Path to the source data file.
* `-o, --output` *(Optional)*: Base name prefix for output files.
* `--channels` *(Optional)*: List of channels to analyze (Default: S1 S2 S3 S4 S5).
* `--error-threshold` *(Optional)*: The target error percentage required to accept an optimal historical window (Default: 1.0).
* `--forecast-row` *(Optional)*: Specific row index to forecast. If left blank, defaults to N+1 (the future).
* `--ensemble-width` *(Optional)*: The +/- variance in window size to use when clustering forecasts around the optimum (Default: 5).
* `--sweep-input` *(Optional)*: Path to a previous sweep file. If provided, the engine will run Dominant Period Detection and optimize its search space based on the resulting period.
* `--min-stack` / `--max-stack`: Bounding constraints for Stack sizes (Defaults: 5 to 41).
* `--min-window` / `--max-window`: Bounding constraints for Window sizes (Defaults: 20 to 151).
* `--batch-mode` *(Optional)*: Enables the highly-optimized Batched Tensor Workflow specifically for the Optimization Phase.
* `--perf [con]` *(Optional)*: Enables deep performance monitoring for the cluster workflow. Adding `con` streams metrics to the terminal.
* `--svd-gpu`: Force SVD calculations to run on the GPU.

**Input File Types & Data Schema:**

* **Main Input** (`--input`): .xlsx or .parquet. The schema must be a continuous, headerless 2D numerical matrix where Rows `=` Time and Columns `=` Variables (mapped positionally by `--channels`), identical to the Base Profiler.
* **Sweep Input** (`--sweep-input`): .parquet or .csv. Must contain the window\_size and {channel}\_err\_pct columns generated by a previous Base Profiler run.

**Operation Output:**

**1. File Output:**

* **File Type**: .json
* **File Naming**: `<output_base>_forecast.json`
* **File Output Schema**: The workflow generates a single structured JSON payload containing the exact configuration used and the statistical results for each channel.

```JSON
{

  "target\_row": 1000,
  "optimal\_window\_center": 142,
  "optimal\_stack": 15,
  "period\_used": 14,
  "ensemble\_count": 11,
  "channels": {
    "S1": {
      "prediction\_median": 452.18,
      "uncertainty\_std": 1.24,
      "actual": 451.90,
      "error": 0.28
    }
  }
}
```

*(Note: actual and error keys are only present if `--forecast-row` targets a known historical row. For future forecasts, they are omitted).*

**2\. Console Output Schema:**

During execution, the tool prints a step-by-step diagnostic trace to the terminal.

* **Initialization**: Target mode status (User Defined vs. Future Forecast) and Ground Truth availability.
* **Step 1**: Optimization: Reports the search space strategy (Full Spectrum vs. Detected Period scaling) and the absolute optimal configuration found, including its exact historical error percentage.
* **Step 2**: Ensemble Forecast: Reports the width of the cluster being processed.
* **Forecast Results**: A clean statistical printout for every requested channel:
  Channel S1: 452.1800 (+/- 1.2400)
  Actual: 451.9000, Error: 0.2800

Example Commands:

```Bash
# Basic cluster forecast for the future (Row N+1)

python main.py cluster -i data.parquet --error-threshold 2.5 --ensemble-width 10
# Smart backtesting using a previously detected period

python main.py cluster -i data.xlsx --forecast-row 500 --sweep-input previous_sweep.parquet
```

### 4. Analysis Tools (Period Detection)

Description: A post-processing diagnostic tool that detects dominant temporal frequencies by analyzing the output of a previous hyperparameter sweep. Instead of analyzing the raw data signal, this tool scans the forecasting error rates (`err_pct`) across all tested window sizes. It identifies cyclical "pulses" in accuracy (local minima, or "needles") to help users pinpoint the absolute optimal Window Sizes for future high-precision DMD forecasts.Workflow:

1. Loads the tabular output file from a previously completed DMD Profiler sweep.
2. Groups the dataset by`window_size` and isolates the absolute minimum prediction error for the target channel.
3. Scans the error curve to identify "needles" (local minima where the error is less than its immediate neighbors and below a 5.0% threshold).
4. Calculates the numerical gaps between these needles to estimate the dominant period.
5. Refines the period and calculates a confidence score based on the consistency of the gap distances.
6. Outputs a formatted diagnostic report to the terminal.

**Command Line Options**: The `analysis period` sub-command utilizes the following arguments:

* `sweep_input`*(Positional, Required)*: The file path to a previously generated sweep results file.
* `-c, --channel`*(Optional)*: The specific data channel you want to analyze for periodic errors. (Default:`S1`).

**Input File Types & Data Schema:**

* **Supported File Types**:`.parquet` or`.csv`
* **Data Schema**: The input file must be the standard tabular output generated by the Base Profiler. The analysis engine strictly requires the file to contain the following columns:
  * `window_size`: The temporal lookback window sizes tested during the sweep.
  * `{channel}_err_pct`: The prediction percentage error column corresponding to your target channel (e.g.,`S1_err_pct`).

**Operation Output:**

* **File Output**: None. This is a read-only post-processing tool designed for quick diagnostics. It does not generate or modify any files.
* **Console Output Schema**: The tool prints a structured report directly to the terminal. Depending on the success of the algorithm, it outputs the following schema:
  * `Status`:`SUCCESS` or`FAILED` (The algorithm fails if it detects fewer than 2 needles/minima).
  * `Reason`:*(Only present if FAILED)* Explains why the analysis aborted.
  * `Dominant Period`: The calculated resonant period interval (in samples/rows).
  * `Confidence`: A percentage (`0.0%` to`100.0%`) indicating how mathematically consistent the gaps between the needles are.
  * `Needles Detected`: The total integer count of valid local minima found on the error curve.
  * `Needle Locations (Windows)`: An array of the exact window sizes where the accuracy spiked (e.g.,`[50, 100, 150]`).
  * `Detected Gaps`: An array of the exact numerical distances between the detected needles.

**Example Command:**

```Bash
python main.py analysis period sweep\_results.parquet \--channel S1
```

---

## Data Input

### Requirements

If you prefer to keep the raw CSV data for alternative downstream processing, you can pass the `--keep-temp` flag at runtime. This instructs the IO Manager to leave the CSV file intact on your hard drive alongside the compiled Parquet/Excel files.

* **Excel (.xlsx):** Read with no headers. Columns are mapped sequentially to requested channels.
* **Parquet (.parquet):** Fast, compressed columnar storage. Preferred for large datasets.

### Input Data Schema & Parameter Mapping

The core mathematical engine treats your input file as a dense, 2D numerical matrix where **Rows \= Time** and **Columns \= Variables**.

**1. Rows (Temporal Snapshots)**

Every row in your dataset must represent a single, sequential snapshot in time. There should be no missing rows or gaps in the time-series.

* The `--start-row` and `--end-row` CLI parameters act as a vertical data slicer. They dictate the exact contiguous block of time-steps the engine will extract into memory to build the Hankel matrices.

**2. Columns (State Variables / Channels)**

Every column represents a distinct feature, sensor, or mathematical state variable. Because the application explicitly expects **headerless** data files, the columns are mapped positionally from left to right.

* The `--channels` argument is mapped sequentially to the column indices (0-indexed) of your input file.
* *Example:* If your dataset has 5 columns, and you pass \--channels Temp Pressure Velocity, the engine maps them exactly in order:
  
  * Column 0: Maps to Temp
  * Column 1: Maps to Pressure
  * Column 2: Maps to Velocity

---

## Data Outputs

The primary forecasting results and error metrics are exported to flat, tabular files (controlled by the \--format argument: parquet, xlsx, or both). The application generates a comprehensive summary file containing the exact predictions and error rates for every single row and hyperparameter combination tested.

Simultaneously, the pure mathematical tensors (SVD, Eigenvalues, DMD Modes) functioning as intermediary elements of the calculations are exported to isolated HDF5 directories.

### 1. Temporary CSV Buffering & The `--keep-temp` Flag

To prevent memory exhaustion during massive parameter sweeps, the application does not hold all results in RAM. Instead, the IO Manager continuously flushes forecast metrics to a temporary CSV file named \<output\_prefix\>\_temp.csv. This file also acts as the vital recovery ledger if you need to use the `--resume` function.

Upon successful completion of the sweep, the application automatically compiles this temporary CSV into your final requested format and deletes it to save disk space. If you prefer to keep the raw CSV data for downstream processing, pass the `--keep-temp` flag at runtime.

#### 2. The Tabular Record Schema

Each row in the output file represents a single prediction generated by a specific Window/Stack combination:

* `data_set_start` / `data_set_end`: The specific historical slice of rows used to build the Hankel matrix for this prediction.
* `data_window_size`: The total number of rows in the historical slice.
* `window_size`: The temporal lookback window ($w$) used.
* `stack_size`: The Time-Delay Embedding depth ($s$) used.
* `rank_ratio`: The truncation threshold used during SVD (Default: 0.99).

**Channel Metrics:** *(Generated for every mapped channel, e.g., `S1`)*
* `{channel}_val_target`: The actual ground-truth value from the dataset at the target row.
* `{channel}_pred_value`: The raw, continuous float prediction calculated by the DMD operator.
* `{channel}_pred_err`: The absolute error ($| \text{target} - \text{prediction} |$).
* `{channel}_err_pct`: The percentage error relative to the target value.

**Integer-Rounded Metrics:**
To support categorical or count-based modeling, the application also generates integer-rounded variants for every continuous metric: 
* `{channel}_val_target_int`, `{channel}_pred_value_int`, `{channel}_pred_err_int`, `{channel}_err_pct_int`.

### 3. Performance Output (_perf.parquet)

If the `--perf` flag is enabled during execution, the application will generate an additional `<output_prefix>_perf.parquet` file alongside your main results. This file contains a highly granular diagnostic log, recording the exact sub-millisecond execution times for every matrix operation mapped to its specific stack size and row iteration.

**The Performance Record Schema:**

Depending on whether you are running the Sequential or Batched workflow, the performance records will contain the following context and timing metrics (all times are recorded in seconds as high-precision floats):

* **Context Metadata:**
  * **target_row / d_start / d_end**: The specific dataset row coordinates and historical boundaries being evaluated.
  * **window_size**: The temporal lookback window ($w$).
  * **stack_size**: The Time-Delay Embedding depth ($s$).
  * direction: The sequential sweep direction (e.g., inc\_start, dec\_end).
* **Micro-Stage Timings:**
  * **stage_1_hankel_s**: Time to construct the time-delay embedded Hankel matrices (X and Y).
  * **stage_2_svd_s**: Time to compute the Truncated Singular Value Decomposition and rank ratio.
  * stage_3_operator_s: Time to compute the Reduced-Order Operator ($\\tilde{A}$) and Eigenvalues.
  * **stage_4_modes_s**: Time to compute the exact spatial DMD Modes ($\\Phi$) and Amplitudes ($b$).
  * stage_5_predict_s: Time to multiply the components and reconstruct the continuous forecast vector.
  * **stage_6_hdf5_s**: Cumulative time spent writing the raw mathematical tensors to the isolated HDF5 directories.
  * **stage_7_format_s**: Time spent formatting the prediction records and error metrics for tabular output.
* **Macro-Level Timings:**
  * **total_stack_s:** Total time to compute the entire DMD pipeline for a single stack size.
  * **total_row_time_s**: (Sequential Mode) Total time to compute all active stack sizes for a single target row.
  * **total_sweep_s**: (Batched Mode) Total time to compute a specific stack size across the entire massive 3D tensor batch.

### 4\. Ultra-Granular HDF5 Output (Schema v2.0.0)

To support downstream Machine Learning workflows, this application utilizes an "Ultra-Granular 1-to-1" HDF5 storage strategy. Instead of dumping all data into a single monolithic file, the engine completely isolates every stage of the mathematical pipeline into its own dedicated directory.

**HDF5 Directory Layout:**

```Plaintext
<output_prefix>_hdf5/  
├── Hankle/                  # Target: `hankle`  
├── SVD_Truncation/          # Target: `svd`  
├── Reduced_Operator/        # Target: `dmd-op`  
├── Eigen/                   # Target: `eigen`  
├── DMD_Modes/               # Target: `dmd-modes`  
├── DMD_Amplitudes/          # Target: `dmd_amp`  
└── Prediction/              # Target: `pred`
```

**HDF5 File Naming Convention:**

When a mathematical tensor is saved, it is placed into a directory corresponding to its Data Class. The file itself is named using the exact hyperparameter coordinates that generated it, using the following schema:

\_ds_[Data Start] \_de_[Data End] \_ws_[Window Start] \_we_[Window End] \_s_[Stack Size].hdf5

*Example:* \_ds\_100\_de\_500\_ws\_350\_we\_500\_s\_20.hdf5 tells you exactly that this file contains the matrices for a dataset evaluated from row 100 to 500, using a lookback window starting at row 350 (a window size of 150), and a stack size of 20\.

### HDF5 Diagnostics & Repair (hdf5 sub-command)

Because large SVD sweeps can generate massive files, the application includes tools to inspect and repair files corrupted by hard crashes or power loss.

**Print the self-describing hierarchical schema of a file:**

```Bash
python main.py hdf5 schema output_svd.hdf5
```

**Inspect a file for truncated B-Trees or corrupt configurations:**

```Bash
python main.py hdf5 inspect output_svd.hdf5
```

**Safely truncate and repair corrupt entries at the tail of a file:**

```Bash
python main.py hdf5 fix output_svd.hdf5
```

---

## Fault Tolerance & Crash Recovery

Massive hyperparameter sweeps can take hours. To prevent data loss from system crashes, the application utilizes a highly robust fault-tolerance mechanism powered by signal handling and JSON state tracking.

### The Graceful Pause & Deadlock-Safe Termination (Ctrl+C)

The application implements custom SIGINT and SIGTERM handling to protect your data and HDF5 integrity:

1. **First Ctrl+C (Pause & Save):** Sets an internal application flag. The math engine is permitted to finish its current in-flight tensor operation. The IO Manager then flushes the temporary CSV buffers to disk, updates the execution state file, and exits cleanly. This is the recommended way to pause a long run.
2. **Second Ctrl+C (Force Kill):** Forces an immediate os.\_exit(1). Use this **only** if a PyTorch C-extension or HDF5 thread has completely deadlocked the Python GIL and is unresponsive to the first interrupt.

### The \--resume Workflow

When a sweep is initiated, the application generates a temporary state file alongside your data buffer.

1. **State Tracking:** After each matrix operation is successfully evaluated, the IOManager updates the state file with the current progress coordinate.
2. **Resumption:** By passing the \--resume flag (e.g., `python main.py --input data.parquet --resume`), the application bypasses standard initialization. It reads the state file, automatically restores your original command-line arguments, safely ignores previously calculated matrices, and picks up exactly where it left off.

*Important Note on Parameter Overrides:* When `--resume` is passed, all other command-line parameters provided at runtime (with the exception of `--help`) are entirely ignored by the application to guarantee state integrity.

---

# Extending the Framework

To add new workflows or algorithms, create a new Controller class in the controllers/ directory and register it in the handlers list within main.py. Keep heavy mathematical operations isolated in the core/ directory to maintain framework independence.

