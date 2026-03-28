import os
import json
import h5py
import numpy as np

# Core Application Constants
SCRIPT_VERSION = "9.3.32"
SCHEMA_VERSION = "2.0.0"  # Restored to granular atomic schema and exact 1-to-1 routing

def build_hierarchical_schema():
    """Defines the 7 strict Data Classes and Data Types for the highly granular HDF5 storage."""
    return {
        "schema_version": SCHEMA_VERSION,
        "global_attributes": {
            "data_set_identifier": {"type": "string", "description": "The Data Class generating this file"},
            "hierarchical_schema": {"type": "string (JSON)", "description": "Full JSON representation of the schema"}
        },
        "data_classes": {
            "Hankle": {
                "H": "float32", "X": "float32", "Y": "float32"
            },
            "SVD_Truncation": {
                "U": "float32", "S": "float32", "Vh": "float32",
                "U_r": "float32", "S_inv": "float32", "V_r": "float32", "r": "int32"
            },
            "Reduced_Operator": {
                "Atilde": "float32"
            },
            "Eigen": {
                "eigvals_real": "float32", "eigvals_imag": "float32",
                "W_eig_real": "float32", "W_eig_imag": "float32"
            },
            "DMD_Modes": {
                "Phi_real": "float32", "Phi_imag": "float32"
            },
            "DMD_Amplitudes": {
                "b_real": "float32", "b_imag": "float32"
            },
            "Prediction": {
                "pred_vec_real": "float32"
            }
        }
    }

def generate_hdf5_dir_name(base_output_name: str) -> str:
    return f"{base_output_name}_svd.hdf5"

def generate_hdf5_file_name(d_start: int, d_end: int, w_start: int, w_end: int, s: int) -> str:
    return f"_ds_{d_start}_de_{d_end}_ws_{w_start}_we_{w_end}_s_{s}.hdf5"

def save_stage_to_hdf5(
    base_dir: str,
    data_class: str,
    d_start: int,
    d_end: int,
    w: int,
    s: int,
    payload: dict,
    full_schema_json: str
):
    """Writes a single mathematical stage's output to its designated atomic file."""
    
    w_start = d_end - w
    w_end = d_end
    
    # 1. Routing & Directory Creation
    class_dir = os.path.join(base_dir, data_class)
    os.makedirs(class_dir, exist_ok=True)
    
    file_name = generate_hdf5_file_name(d_start, d_end, w_start, w_end, s)
    file_path = os.path.join(class_dir, file_name)

    # 2. Metadata Preparation
    hdf5_entry_metadata = {
        "w_start": w_start,
        "w_end": w_end,
        "stack_size": s,
        "data_class": data_class,
        "data_types": list(payload.keys())
    }

    group_name = f"{data_class}_ds_{d_start}_de_{d_end}_ws_{w_start}_we_{w_end}_s_{s}"

    # 3. Atomic Write & Flush (with scalar safety check)
    # Using 'w' to guarantee a fresh, isolated file per hyperparameter combination
    with h5py.File(file_path, "w") as hf:
        hf.attrs["data_set_identifier"] = data_class
        hf.attrs["hierarchical_schema"] = full_schema_json
        
        grp = hf.create_group(group_name)
        for k, v in hdf5_entry_metadata.items():
            grp.attrs[k] = v
            
        for data_type, data_value in payload.items():
            if data_value is not None:
                val_arr = np.asarray(data_value)
                if val_arr.ndim == 0:
                    # Save scalars (like 'r') WITHOUT compression
                    grp.create_dataset(data_type, data=data_value)
                else:
                    # Save matrices WITH compression
                    grp.create_dataset(data_type, data=data_value, compression="gzip", compression_opts=4)
                
        hf.flush()


# --- LEGACY HDF5 DIAGNOSTICS FOR CLI COMPATIBILITY ---
def analyze_and_fix_hdf5(file_path: str, fix: bool = False, abort_check=None) -> None:
    print(f"\n=== HDF5 Diagnostics & Repair Tool (v{SCRIPT_VERSION}) ===")
    print(f"Target File: {file_path}")
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return
    print("[Info] Note: Legacy monolithic repair tool. Atomic directory-based runs do not require tail truncation.")

def print_hdf5_schema(file_path: str) -> None:
    print(f"\n=== HDF5 Hierarchical Schema Inspector (v{SCRIPT_VERSION}) ===")
    print(f"Target File/Directory: {file_path}")
    if not os.path.exists(file_path):
        print(f"[Error] Target not found: {file_path}")
        return
        
    target_h5 = file_path
    if os.path.isdir(file_path):
        for root, _, files in os.walk(file_path):
            for file in files:
                if file.endswith(".hdf5"):
                    target_h5 = os.path.join(root, file)
                    break

    try:
        with h5py.File(target_h5, "r") as hf:
            if "hierarchical_schema" in hf.attrs:
                schema = json.loads(hf.attrs["hierarchical_schema"])
                print("\n--- 1. Global File Attributes ---")
                for key, val in hf.attrs.items():
                    if key not in ["hierarchical_schema"]:
                        print(f"  - {key}: {val}")
                print("\n--- Embedded Hierarchical Schema Definition ---")
                print(json.dumps(schema, indent=2))
            else:
                print("\n[Info] Hierarchical schema missing. Legacy file detected.")
    except Exception as e:
        print(f"[Error] Schema inspection failed: {e}")