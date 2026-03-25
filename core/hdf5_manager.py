
import os
import json
import h5py

# Core Application Constants
SCRIPT_VERSION = "9.3.32"
SCHEMA_VERSION = "1.2.0"

def build_hierarchical_schema(store_dmd=False, store_eigen=False, no_modes=False, store_hankel=False):
    """
    Dynamically constructs the self-describing JSON schema based on active configuration.
    Injects data types and operational descriptions for every element.
    """
    schema = {
        "schema_version": SCHEMA_VERSION,
        "global_attributes": {
            "schema_version": {"type": "string", "description": "Version identifier of the structural schema"},
            "script_version": {"type": "string", "description": "Version of the script that initialized the file"},
            "input_file": {"type": "string", "description": "Name of the source data file analyzed"},
            "channels": {"type": "string (JSON)", "description": "Array of channels analyzed in the dataset"},
            "data_sets": {"type": "string (JSON)", "description": "Flat list of all mathematical dataset arrays resident in the HDF5 file"},
            "hierarchical_schema": {"type": "string (JSON)", "description": "This complete, self-describing structural model"}
        },
        "representative_group_schema": {
            "group_attributes": {
                "window_size": {"type": "int", "description": "Temporal length of the evaluated DMD window (m)"},
                "stack_size": {"type": "int", "description": "Number of time-delay embedding stacks used"},
                "data_set_start": {"type": "int", "description": "Starting row index of the data slice"},
                "data_set_end": {"type": "int", "description": "Ending row index of the data slice"},
                "data_window_size": {"type": "int", "description": "Total length of the data slice evaluated"},
                "record_type": {"type": "string", "description": "Phase of the algorithm workflow (opt, forecast, etc.)"},
                "rank_used": {"type": "int", "description": "Truncated integer rank (r) used for SVD"},
                "rank_ratio": {"type": "float", "description": "Threshold ratio used to determine the dynamic rank"}
            },
            "datasets": {
                "svd_data": {
                    "U": {"type": "float32", "description": "Left singular vectors mapping spatial/stacked features"},
                    "S": {"type": "float32", "description": "Singular values representing structural variance"},
                    "Vh": {"type": "float32", "description": "Right singular vectors mapping temporal evolution"}
                },
                "predictions": {
                    "pred_vec_real": {"type": "float32", "description": "Real part of the forecasted/reconstructed state vector"}
                }
            }
        }
    }
    
    datasets = schema["representative_group_schema"]["datasets"]
    
    # Conditional DMD Operator & Eigen schema
    if store_dmd or store_eigen:
        datasets["dmd_operator_and_eigen"] = {}
        if store_dmd:
            datasets["dmd_operator_and_eigen"]["Atilde"] = {"type": "float32", "description": "Reduced-order DMD operator matrix (Atilde)"}
        if store_eigen:
            datasets["dmd_operator_and_eigen"].update({
                "eigvals_real": {"type": "float32", "description": "Real part of the operator's eigenvalues"},
                "eigvals_imag": {"type": "float32", "description": "Imaginary part of the operator's eigenvalues"},
                "W_eig_real": {"type": "float32", "description": "Real part of the eigenvectors of Atilde"},
                "W_eig_imag": {"type": "float32", "description": "Imaginary part of the eigenvectors of Atilde"}
            })
            
    # Conditional Modes & Amplitudes schema
    if store_eigen and not no_modes:
        datasets["dmd_modes_and_amplitudes"] = {
            "Phi_real": {"type": "float32", "description": "Real part of the exact DMD modes"},
            "Phi_imag": {"type": "float32", "description": "Imaginary part of the exact DMD modes"},
            "b_real": {"type": "float32", "description": "Real part of the initial mode amplitudes"},
            "b_imag": {"type": "float32", "description": "Imaginary part of the initial mode amplitudes"}
        }
        
    # Conditional Raw Hankel schema
    if store_hankel:
        datasets["hankel_matrix"] = {
            "H": {"type": "float32", "description": "Full time-delay embedded Hankel matrix"}
        }
        
    return schema

def save_to_hdf5(
    hf,
    group_name: str,
    result: dict,
    metadata: dict,
    store_hankel: bool = False,
    no_modes: bool = False,
    store_dmd: bool = False,
    store_eigen: bool = False,
) -> None:
    """Core logic for writing DMD SVD results to the HDF5 hierarchical structure."""
    if group_name in hf:
        return

    grp = hf.create_group(group_name)

    # 1. Base Attributes
    for key, val in metadata.items():
        if val is not None:
            grp.attrs[key] = val
    if "rank_used" in result:
        grp.attrs["rank_used"] = int(result["rank_used"])
    if "rank_ratio" in result:
        grp.attrs["rank_ratio"] = float(result["rank_ratio"])

    # 2. SVD Data Subgroup
    svd_grp = grp.create_group("svd_data")
    for k in ["U", "S", "Vh"]:
        if result.get(k) is not None:
            svd_grp.create_dataset(k, data=result[k], compression="gzip", compression_opts=4)

    # 3. Predictions Subgroup
    pred_grp = grp.create_group("predictions")
    if result.get("pred_vec_real") is not None:
        pred_grp.create_dataset("pred_vec_real", data=result["pred_vec_real"], compression="gzip", compression_opts=4)

    # 4. DMD Operator and Eigendecomposition Subgroup
    if store_dmd or store_eigen:
        dmd_eigen_grp = grp.create_group("dmd_operator_and_eigen")
        if store_dmd and result.get("Atilde") is not None:
            dmd_eigen_grp.create_dataset("Atilde", data=result["Atilde"], compression="gzip", compression_opts=4)
        if store_eigen:
            for k in ["eigvals_real", "eigvals_imag", "W_eig_real", "W_eig_imag"]:
                if result.get(k) is not None:
                    dmd_eigen_grp.create_dataset(k, data=result[k], compression="gzip", compression_opts=4)

    # 5. DMD Modes and Amplitudes Subgroup
    if store_eigen and not no_modes:
        modes_grp = grp.create_group("dmd_modes_and_amplitudes")
        for k in ["Phi_real", "Phi_imag", "b_real", "b_imag"]:
            if result.get(k) is not None:
                modes_grp.create_dataset(k, data=result[k], compression="gzip", compression_opts=4)

    # 6. Raw Reshaped Data Subgroup (Hankel Matrix)
    if store_hankel and result.get("H") is not None:
        raw_grp = grp.create_group("hankel_matrix")
        raw_grp.create_dataset("H", data=result["H"], compression="gzip", compression_opts=4)

def analyze_and_fix_hdf5(file_path: str, fix: bool = False, abort_check=None) -> None:
    """
    Analyzes an HDF5 file backwards for incomplete/corrupt configuration groups.
    Tolerates truncated/damaged internal B-trees and addr overflows.
    """
    print(f"\n=== HDF5 Diagnostics & Repair Tool (v{SCRIPT_VERSION}) ===")
    print(f"Target File: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    mode = "r+" if fix else "r"
    try:
        with h5py.File(file_path, mode) as hf:
            group_names = []
            
            # Explicit and fault-tolerant iterative key scan
            try:
                keys_iter = iter(hf.keys())
                while True:
                    if abort_check and abort_check():
                        print("\n[Operation Aborted] User interrupted the file index scan.")
                        return
                    try:
                        k = next(keys_iter)
                        group_names.append(k)
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"\n[Warning] HDF5 tree truncation encountered: {e}")
                        print(f"          Recovered {len(group_names)} group links before the corruption boundary.")
                        break
            except Exception as e:
                print(f"[Error] Failed to initialize key iterator. File header may be deeply corrupted: {e}")
                return
                
            group_names.sort()
            
            if not group_names:
                print("\n[Result] No valid configuration groups found in the file.")
                return

            file_schema_version = hf.attrs.get("schema_version", "Unknown (Legacy)")
            print(f"Schema Version Identifier: {file_schema_version}")
            
            expected_attrs = set()
            expected_datasets = set()
            
            if "hierarchical_schema" in hf.attrs:
                print("[Info] Hierarchical schema located. Extracting direct validation contract...")
                schema = json.loads(hf.attrs["hierarchical_schema"])
                
                expected_attrs = set(schema["representative_group_schema"]["group_attributes"].keys())
                for category, dsets in schema["representative_group_schema"]["datasets"].items():
                    for ds_name in dsets.keys():
                        expected_datasets.add(f"{category}/{ds_name}")
            else:
                print("[Info] Hierarchical schema missing. Using legacy baseline fallback.")
                expected_attrs = {"window_size", "stack_size", "data_set_start", "data_set_end"}
                expected_datasets = {"U", "S", "Vh", "pred_vec_real"}
                
            print(f"       Required Datasets for Validation: {list(expected_datasets)}")

            last_good_group = None
            last_good_attrs = {}
            corrupt_groups = []

            # Iterate backwards to find the most recent valid state
            for g_name in reversed(group_names):
                if abort_check and abort_check():
                    print("\n[Operation Aborted] User interrupted the diagnostic analysis.")
                    return
                    
                try:
                    grp = hf[g_name]
                    if not isinstance(grp, h5py.Group):
                        continue
                        
                    has_attrs = expected_attrs.issubset(grp.attrs.keys())
                    has_datasets = all(ds_path in grp for ds_path in expected_datasets)
                    
                    if has_attrs and has_datasets:
                        last_good_group = g_name
                        last_good_attrs = {k: grp.attrs[k] for k in grp.attrs.keys()}
                        break
                    else:
                        corrupt_groups.append(g_name)
                except Exception:
                    # Catches 'addr overflow' or broken node faults during array access
                    corrupt_groups.append(g_name)

            if fix:
                if corrupt_groups:
                    print(f"\nAction: Found {len(corrupt_groups)} corrupt/incomplete entries at the tail. Truncating...")
                    for g_name in corrupt_groups:
                        if abort_check and abort_check():
                            print("\n[Operation Aborted] File truncation safely paused.")
                            return
                        try:
                            del hf[g_name]
                        except Exception as e:
                            print(f"  [Warning] Could not cleanly delete unlinked node '{g_name}': {e}")
                    print("Status: Truncation complete. File repaired.")
                else:
                    print("\nStatus: No corrupt entries found. File is clean.")
            else:
                if corrupt_groups:
                    print(f"\n[Warning] Found {len(corrupt_groups)} corrupt/incomplete entries at the tail of the file.")
                else:
                    print("\nStatus: No corrupt entries found. File is clean.")

            if last_good_group:
                print("\n--- Last Good Hyperparameters ---")
                print(f"group_name: {last_good_group}")
                for key, val in last_good_attrs.items():
                    print(f"{key}: {val}")
                print("---------------------------------")
            else:
                print("\n[Result] All configuration groups in the file appear to be corrupt based on schema.")
                
    except OSError as e:
        print(f"[Error] Failed to open or read the HDF5 file (it may be severely corrupted): {e}")
    except Exception as e:
        print(f"[Error] An unexpected error occurred during HDF5 analysis: {e}")

def print_hdf5_schema(file_path: str) -> None:
    """Opens the target HDF5 file and explicitly prints out the hierarchical schema."""
    print(f"\n=== HDF5 Hierarchical Schema Inspector (v{SCRIPT_VERSION}) ===")
    print(f"Target File: {file_path}")

    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    try:
        with h5py.File(file_path, "r") as hf:
            if "hierarchical_schema" in hf.attrs:
                schema = json.loads(hf.attrs["hierarchical_schema"])
                print("\n--- 1. Global File Attributes ---")
                for key, val in hf.attrs.items():
                    if key not in ["hierarchical_schema", "schema_datasets", "schema_attributes", "data_sets"]:
                        print(f"  - {key}: {val}")
                if "data_sets" in hf.attrs:
                    print(f"  - data_sets: {json.loads(hf.attrs['data_sets'])}")
                    
                print("\n--- Embedded Hierarchical Schema Definition ---")
                print(json.dumps(schema, indent=2))
                print("\n===================================================\n")
            else:
                print("\n[Info] Hierarchical schema missing from global attributes. Legacy file detected.")
    except Exception as e:
        print(f"[Error] An unexpected error occurred during schema inspection: {e}")