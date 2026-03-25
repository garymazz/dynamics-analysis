# DMD Tool (v9.3.32)

A high-performance, GPU-accelerated framework for time-series forecasting and structural analysis using Dynamic Mode Decomposition (DMD).

Built on the **Cement CLI Framework** and **PyTorch**, this application provides a robust suite of tools for sweeping DMD hyperparameters, detecting dominant temporal periods, running smart ensemble forecasts, and managing large-scale matrix decompositions via fault-tolerant HDF5 storage.

## Key Features

* **GPU-Accelerated Math Engine:** Leverages PyTorch (`torch.linalg.svd`) to process massive time-delay embedded Hankel matrices across multiple channels simultaneously.
* **Modular Architecture:** Built on the Cement framework, ensuring clean separation of concerns between CLI routing, Pandas I/O, HDF5 management, and pure mathematical operations.
* **Cluster-DMD Forecasting:** A "smart mode" workflow that automatically scans historical data for optimal reconstruction windows based on dominant period detection, generating ensemble forecasts with uncertainty metrics.
* **Fault-Tolerant HDF5 Storage:** Writes exact DMD operators, spatial modes, and temporal dynamics to self-describing, hierarchically structured HDF5 files. Includes custom B-tree repair tools to recover data from interrupted runs.
* **Deadlock-Safe Termination:** Implements a two-stage signal interception mechanism allowing graceful shutdowns during heavy C-extension/GPU workloads, preventing data corruption.

---

## System Architecture

The codebase has been decoupled from a monolithic script into a scalable MVC-like structure using Cement:


dmd_profiler/
├── main.py                 # Application bootstrap and global signal handlers
├── core/                   # Framework-agnostic business logic
│   ├── math_engine.py      # Pure PyTorch SVD/DMD tensor mathematics
│   ├── io_manager.py       # Pandas data loading, buffering, and Parquet/Excel exporting
│   ├── hdf5_manager.py     # HDF5 schema generation, injection, and diagnostics
│   └── period_analysis.py  # Algorithms for dominant period gap detection
└── controllers/            # Cement CLI Routing and Argument Parsing
    ├── base.py             # Default execution (Standard Sweeps & Cluster-DMD)
    ├── hdf5_tools.py       # Sub-commands for HDF5 repair and inspection
    └── analysis_tools.py   # Sub-commands for standalone period analysis


## Installation

**Prerequisites**

* Python 3.8+
* CUDA Toolkit (Optional, but highly recommended for PyTorch GPU acceleration)

**Dependencies**
Install the required packages via pip:

**Bash**
`pip install cement torch numpy pandas h5py openpyxl pyarrow`

*(Note: openpyxl is required for reading/writing .xlsx files; pyarrow is required for .parquet files).*

## Usage Guide

The application utilizes a Git-like command-line interface with nested sub-commands.


### Standard Parameter Sweeps (Base Controller)

The default execution mode runs heavy matrix sweeps across a range of window and stack sizes.

**#### Basic Forwards Sweep (Increasing Start Row):**

Bash
`python main.py --input data.xlsx --inc-start --start-row 1 --end-row 500 --min-stack 5 --max-stack 20`

**#### Basic Backwards Sweep (Decreasing End Row):**

**Bash**
`python main.py --input data.parquet --dec-end --min-window 20 --max-window 150 --hdf5-all`

**Key Arguments:**

* -i, --input: Source data file (.xlsx or .parquet).
* --channels: Channels to analyze (Default: S1 S2 S3 S4 S5).

* --hdf5-all: Save full Eigendecomposition and DMD Operators to the output HDF5 file.

* --svd-gpu: Force SVD calculations to run on the GPU if available.

### Cluster-DMD Forecasting

Enable the "Smart Mode" to automatically optimize window/stack sizes based on historical error thresholds and generate future predictions.

**Bash**
`python main.py --input data.xlsx --cluster-dmd --forecast-row 1000 --min-window 50 --max-window 200`

### HDF5 Diagnostics & Repair (hdf5 sub-command)

Because large SVD sweeps can generate massive HDF5 files, the application includes tools to inspect and repair files corrupted by hard crashes or power loss.

#### Print the self-describing hierarchical schema of a file:

**Bash**
`python main.py hdf5 schema output_svd.hdf5`

### Inspect a file for truncated B-Trees or corrupt configurations:

**Bash**
`python main.py hdf5 inspect output_svd.hdf5`

#### Safely truncate and repair corrupt entries at the tail of a file:

**Bash**
`python main.py hdf5 fix output_svd.hdf5`

#### Post-Processing Analysis (analysis sub-command)

Run standalone analysis on generated sweep data.

**Detect dominant temporal periods from a previous parameter sweep:**

Bash
`python main.py analysis period sweep_results.parquet --channel S1`

## Data Input Requirements

The application expects continuous time-series data without headers.

* Excel (.xlsx): Read with no headers. Columns are mapped sequentially to requested channels (e.g., Column 0 is S1, Column 1 is S2).
* Parquet (.parquet): Fast, compressed columnar storage. Preferred for large datasets.

## Graceful Termination

The application implements custom SIGINT and SIGTERM handling to protect your data during long sweeps:

1. First`Ctrl+C`: Sets an internal application flag. The math engine will finish its current tensor operation, write the buffers safely to disk, and exit cleanly.
1. Second`Ctrl+C`: Forces an immediate`os._exit(1)`. Use this only if a PyTorch C-extension or HDF5 thread has completely deadlocked the Python GIL and is unresponsive to the first interrupt.

## Extending the Framework

To add new workflows or algorithms, create a new Controller class in the controllers/ directory and register it in the handlers list within main.py. Keep heavy mathematical operations isolated in the core/ directory to maintain framework independence.
