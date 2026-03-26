import torch
import numpy as np

# Device Configuration evaluated at module load
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# STAGES 1-5: PURE MATHEMATICAL OPERATIONS
# ==========================================

def build_hankel_matrix(local_data: torch.Tensor, w: int, s: int) -> tuple:
    """Stage 1: Expands the raw time-series tensor into the augmented Hankel state space."""
    num_channels = local_data.shape[1]
    channels_unfolded = []
    
    for c in range(num_channels):
        if local_data.shape[0] < w:
            return None, None, None
        channels_unfolded.append(local_data[:, c].unfold(0, s, 1).T)
        
    if not channels_unfolded:
        return None, None, None

    H = torch.cat(channels_unfolded, dim=0)
    X = H[:, :-1]
    Y = H[:, 1:]
    
    return H, X, Y

def compute_svd(X: torch.Tensor, svd_gpu: bool, device: str) -> tuple:
    """Stage 2: Executes the SVD, handles device bridging, and truncates to 99% variance."""
    X_svd = X.to("cuda") if (svd_gpu and device == "cuda") else X

    try:
        U_svd, S_svd, Vh_svd = torch.linalg.svd(X_svd, full_matrices=False)
    except Exception:
        return None

    # Bridge back to active working device
    if X_svd.device.type == "cuda" and device == "cpu":
        U, S, Vh = U_svd.to("cpu"), S_svd.to("cpu"), Vh_svd.to("cpu")
    else:
        U, S, Vh = U_svd.to(device), S_svd.to(device), Vh_svd.to(device)

    # 99% Variance Truncation
    max_rank = len(S)
    r = max(1, int(max_rank * 0.99))
    U_r = U[:, :r]
    S_inv = torch.diag(1.0 / S[:r])
    V_r = Vh[:r, :]
    
    return U, S, Vh, U_r, S_inv, V_r, r

def compute_dmd_operator(U_r: torch.Tensor, Y: torch.Tensor, V_r: torch.Tensor, S_inv: torch.Tensor) -> tuple:
    """Stage 3: Calculates the reduced dynamics operator (Atilde) and its eigendecomposition."""
    Atilde = U_r.T @ Y @ V_r.T @ S_inv
    eigvals, W_eig = torch.linalg.eig(Atilde)
    return Atilde, eigvals, W_eig

def compute_dmd_modes(Y: torch.Tensor, V_r: torch.Tensor, S_inv: torch.Tensor, W_eig: torch.Tensor, x_last: torch.Tensor) -> tuple:
    """Stage 4: Reconstructs the high-dimensional spatial modes (Phi) and initial weights (b)."""
    # Complex casting required for spatial mode reconstruction
    Y_c = Y.to(torch.complex64)
    V_r_c = V_r.T.to(torch.complex64)
    S_inv_c = S_inv.to(torch.complex64)

    Phi = Y_c @ V_r_c @ S_inv_c @ W_eig
    x_last_c = x_last.to(torch.complex64)
    
    b = torch.linalg.pinv(Phi) @ x_last_c
    return Phi, b

def reconstruct_and_predict(Phi: torch.Tensor, eigvals: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stage 5: Projects the dynamics forward in time to generate forecasted values."""
    pred_vec = Phi @ torch.diag(eigvals) @ b
    return pred_vec.real

# ==========================================
# STAGE 6: DATA FORMATTING & OUTPUT
# ==========================================

def format_record(pred_vec_real: torch.Tensor, target_vals, channel_names: list, d_start: int, d_end: int, w: int, s: int, rec_type: str) -> dict:
    """Stage 6: Parses the continuous predicted vector back into discrete channel forecasts and errors."""
    record = {
        "type": rec_type,
        "data_set_start": d_start,
        "data_set_end": d_end,
        "data_window_size": d_end - d_start,
        "window_size": w,
        "stack_size": s,
        "rank_ratio": 0.99,
    }

    for i, ch in enumerate(channel_names):
        idx = (i + 1) * s - 1
        if idx >= len(pred_vec_real):
            continue

        pred_val = float(pred_vec_real[idx].item())
        p_int = int(round(pred_val))

        record[f"{ch}_pred_value"] = pred_val
        record[f"{ch}_pred_value_int"] = p_int

        if target_vals is not None:
            tgt = float(target_vals[i])
            t_int = int(tgt)
            err = abs(pred_val - tgt)
            pct = (err / max(1.0, abs(tgt))) * 100.0

            record[f"{ch}_val_target"] = tgt
            record[f"{ch}_pred_err"] = err
            record[f"{ch}_err_pct"] = pct
            record[f"{ch}_val_target_int"] = t_int
            record[f"{ch}_pred_err_int"] = abs(p_int - t_int)
            record[f"{ch}_err_pct_int"] = (abs(p_int - t_int) / max(1.0, abs(t_int))) * 100.0
        else:
            for suffix in ["val_target", "pred_err", "err_pct", "val_target_int", "pred_err_int", "err_pct_int"]:
                record[f"{ch}_{suffix}"] = None

    return record


# ==========================================
# ORCHESTRATORS
# ==========================================

def process_window_group(
    local_data,
    rec_type,
    target_vals,
    d_start,
    d_end,
    w,
    stack_list,
    channel_names,
    abort_check=None,
    record_callback=None,
    hdf5_callback=None,
    svd_store_hankel=False,
    svd_no_modes=False,
    svd_gpu=False,
    hdf5_dmd=False,
    hdf5_eigen=False,
):
    """
    Core PyTorch DMD orchestration engine.
    Sequentially executes the mathematical stages and handles data callbacks.
    """
    results = []

    for s in stack_list:
        if abort_check and abort_check():
            break

        # Stage 1: Embed
        H, X, Y = build_hankel_matrix(local_data, w, s)
        if H is None: continue

        # Stage 2: SVD
        svd_res = compute_svd(X, svd_gpu, DEVICE)
        if svd_res is None: continue
        U, S, Vh, U_r, S_inv, V_r, r = svd_res

        # Stage 3: Operator
        Atilde, eigvals, W_eig = compute_dmd_operator(U_r, Y, V_r, S_inv)

        # Stage 4: Modes
        x_last = X[:, -1]
        Phi, b = compute_dmd_modes(Y, V_r, S_inv, W_eig, x_last)

        # Stage 5: Predict
        pred_vec_real = reconstruct_and_predict(Phi, eigvals, b)

        # Stage 6a: HDF5 Framework Callback
        if hdf5_callback is not None:
            try:
                svd_result = {
                    "U": U.detach().cpu().numpy().astype(np.float32),
                    "S": S.detach().cpu().numpy().astype(np.float32),
                    "Vh": Vh.detach().cpu().numpy().astype(np.float32),
                    "Atilde": Atilde.detach().cpu().numpy().astype(np.float32),
                    "eigvals_real": eigvals.real.detach().cpu().numpy().astype(np.float32),
                    "eigvals_imag": eigvals.imag.detach().cpu().numpy().astype(np.float32),
                    "W_eig_real": W_eig.real.detach().cpu().numpy().astype(np.float32),
                    "W_eig_imag": W_eig.imag.detach().cpu().numpy().astype(np.float32),
                    "Phi_real": Phi.real.detach().cpu().numpy().astype(np.float32),
                    "Phi_imag": Phi.imag.detach().cpu().numpy().astype(np.float32),
                    "b_real": b.real.detach().cpu().numpy().astype(np.float32),
                    "b_imag": b.imag.detach().cpu().numpy().astype(np.float32),
                    "pred_vec_real": pred_vec_real.detach().cpu().numpy().astype(np.float32),
                    "H": H.detach().cpu().numpy().astype(np.float32) if svd_store_hankel else None,
                    "rank_used": r,
                    "rank_ratio": 0.99,
                }

                group_name = f"d{d_start:05d}_e{d_end:05d}_w{w:04d}_s{s:03d}"
                svd_metadata = {
                    "data_set_start": d_start,
                    "data_set_end": d_end,
                    "data_window_size": d_end - d_start,
                    "window_size": w,
                    "stack_size": s,
                    "record_type": rec_type,
                }

                hdf5_callback(
                    group_name=group_name,
                    result=svd_result,
                    metadata=svd_metadata,
                    store_hankel=svd_store_hankel,
                    no_modes=svd_no_modes,
                    store_dmd=hdf5_dmd,
                    store_eigen=hdf5_eigen,
                )
            except Exception as e:
                print(f"Warning: Failed to execute HDF5 callback for group {d_start},{d_end},{w},{s}: {e}")

        # Stage 6b: Tabular DataFrame Callback
        record = format_record(pred_vec_real, target_vals, channel_names, d_start, d_end, w, s, rec_type)
        results.append(record)
        
        if record_callback:
            record_callback([record])

    return results

def run_sweeps_gpu_grouped(
    full_data_matrix,
    dataset_start_idx,
    dataset_end_idx,
    verification_vals,
    enable_train_rec,
    channel_names,
    stack_range,
    abort_check=None,
    record_callback=None,
    hdf5_callback=None,
    svd_store_hankel=False,
    svd_no_modes=False,
    svd_gpu=False,
    hdf5_dmd=False,
    hdf5_eigen=False,
):
    """Single-window DMD wrapper for sweeping stack sizes."""
    train_full = torch.tensor(full_data_matrix, device=DEVICE, dtype=torch.float32)
    data_len = train_full.shape[0]

    w = data_len
    if data_len < 2:
        return []

    val_train = train_full[:-1]
    val_targets = train_full[-1, :].cpu().numpy() if enable_train_rec and data_len > 1 else None
    real_targets = verification_vals

    valid_stacks = [s for s in stack_range if s < w and (w - s + 1) >= 2]
    if not valid_stacks:
        return []

    all_results = []

    if enable_train_rec and val_targets is not None and val_train.shape[0] >= w:
        local_data_val = val_train[-w:, :]
        res = process_window_group(
            local_data=local_data_val,
            rec_type="train_rec",
            target_vals=val_targets,
            d_start=dataset_start_idx,
            d_end=dataset_end_idx,
            w=w,
            stack_list=valid_stacks,
            channel_names=channel_names,
            abort_check=abort_check,
            record_callback=record_callback,
            hdf5_callback=hdf5_callback,
            svd_store_hankel=svd_store_hankel,
            svd_no_modes=svd_no_modes,
            svd_gpu=svd_gpu,
            hdf5_dmd=hdf5_dmd,
            hdf5_eigen=hdf5_eigen,
        )
        all_results.extend(res)

    local_data_real = train_full[-w:, :]
    res = process_window_group(
        local_data=local_data_real,
        rec_type="pred_rec",
        target_vals=real_targets,
        d_start=dataset_start_idx,
        d_end=dataset_end_idx,
        w=w,
        stack_list=valid_stacks,
        channel_names=channel_names,
        abort_check=abort_check,
        record_callback=record_callback,
        hdf5_callback=hdf5_callback,
        svd_store_hankel=svd_store_hankel,
        svd_no_modes=svd_no_modes,
        svd_gpu=svd_gpu,
        hdf5_dmd=hdf5_dmd,
        hdf5_eigen=hdf5_eigen,
    )
    all_results.extend(res)
    
    return all_results