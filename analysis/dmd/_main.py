import time
import torch
import numpy as np

# Device Configuration evaluated at module load
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# INDEPENDENT QUANTITATIVE ANALYSIS
# ==========================================

def calculate_prediction_errors(pred_val: float, target_val: float) -> dict:
    """Quantitative analysis: Computes continuous and integer-based error metrics."""
    if target_val is None:
        return {}
        
    p_int = int(round(pred_val))
    t_int = int(target_val)
    err = abs(pred_val - target_val)
    pct = (err / max(1.0, abs(target_val))) * 100.0
    err_int = abs(p_int - t_int)
    pct_int = (err_int / max(1.0, abs(t_int))) * 100.0
    
    return {
        "val_target": float(target_val),
        "pred_err": err,
        "err_pct": pct,
        "val_target_int": t_int,
        "pred_err_int": err_int,
        "err_pct_int": pct_int
    }

# ==========================================
# STAGES 1-5: PURE MATHEMATICAL OPERATIONS
# ==========================================

def build_hankel_matrix(local_data: torch.Tensor, w: int, s: int) -> tuple:
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
    X_svd = X.to("cuda") if (svd_gpu and device == "cuda") else X
    try:
        U_svd, S_svd, Vh_svd = torch.linalg.svd(X_svd, full_matrices=False)
    except Exception:
        return None

    if X_svd.device.type == "cuda" and device == "cpu":
        U, S, Vh = U_svd.to("cpu"), S_svd.to("cpu"), Vh_svd.to("cpu")
    else:
        U, S, Vh = U_svd.to(device), S_svd.to(device), Vh_svd.to(device)

    max_rank = len(S)
    r = max(1, int(max_rank * 0.99))
    U_r = U[:, :r]
    S_inv = torch.diag(1.0 / S[:r])
    V_r = Vh[:r, :]
    
    return U, S, Vh, U_r, S_inv, V_r, r

def compute_dmd_operator(U_r: torch.Tensor, Y: torch.Tensor, V_r: torch.Tensor, S_inv: torch.Tensor) -> tuple:
    Atilde = U_r.T @ Y @ V_r.T @ S_inv
    eigvals, W_eig = torch.linalg.eig(Atilde)
    return Atilde, eigvals, W_eig

def compute_dmd_modes(Y: torch.Tensor, V_r: torch.Tensor, S_inv: torch.Tensor, W_eig: torch.Tensor, x_last: torch.Tensor, dmd_lstsq: bool = False) -> tuple:
    Y_c = Y.to(torch.complex64)
    V_r_c = V_r.T.to(torch.complex64)
    S_inv_c = S_inv.to(torch.complex64)
    Phi = Y_c @ V_r_c @ S_inv_c @ W_eig
    x_last_c = x_last.to(torch.complex64)
    
    if dmd_lstsq:
        # direct Least-Squares solver instead of Pseudo-Inverse
        b = torch.linalg.lstsq(Phi, x_last_c).solution
    else:
        b = torch.linalg.pinv(Phi) @ x_last_c
    return Phi, b

def reconstruct_and_predict(Phi: torch.Tensor, eigvals: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pred_vec = Phi @ torch.diag(eigvals) @ b
    return pred_vec.real

# ==========================================
# STAGE 6: DATA FORMATTING & OUTPUT
# ==========================================

def format_record(pred_vec_real: torch.Tensor, target_vals, channel_names: list, d_start: int, d_end: int, w: int, s: int, rec_type: str) -> dict:
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
        record[f"{ch}_pred_value"] = pred_val
        record[f"{ch}_pred_value_int"] = int(round(pred_val))

        if target_vals is not None:
            err_metrics = calculate_prediction_errors(pred_val, float(target_vals[i]))
            for key, val in err_metrics.items():
                record[f"{ch}_{key}"] = val
        else:
            for suffix in ["val_target", "pred_err", "err_pct", "val_target_int", "pred_err_int", "err_pct_int"]:
                record[f"{ch}_{suffix}"] = None

    return record


# ==========================================
# SEQUENTIAL ORCHESTRATORS 
# ==========================================

def process_window_group(
    local_data, rec_type, target_vals, d_start, d_end, w, stack_list, channel_names,
    abort_check=None, record_callback=None, hdf5_callback=None, svd_gpu=False, hdf5_targets=None, dmd_lstsq=False, perf_mode=None
):
    results = []
    perf_records = []
    
    # Pre-calculate active HDF5 targets
    hdf5_targets = hdf5_targets or []
    write_all = "all" in hdf5_targets

    for s in stack_list:
        if abort_check and abort_check():
            break
            
        sweep_start_time = time.perf_counter()
        perf_record = {
            "stack_size": s, "window_size": w, "rec_type": rec_type, 
            "d_start": d_start, "d_end": d_end, "stage_6_hdf5_s": 0.0
        }

        # --- Stage 1: Embed ---
        stage_start = time.perf_counter()
        H, X, Y = build_hankel_matrix(local_data, w, s)
        perf_record["stage_1_hankel_s"] = time.perf_counter() - stage_start
        
        if H is None: continue
        
        if hdf5_callback and ("hankle" in hdf5_targets or write_all):
            h_start = time.perf_counter()
            hdf5_callback("Hankle", d_start, d_end, w, s, {
                "H": H.detach().cpu().numpy().astype(np.float32),
                "X": X.detach().cpu().numpy().astype(np.float32),
                "Y": Y.detach().cpu().numpy().astype(np.float32)
            })
            perf_record["stage_6_hdf5_s"] += time.perf_counter() - h_start

        # --- Stage 2: SVD ---
        stage_start = time.perf_counter()
        svd_res = compute_svd(X, svd_gpu, DEVICE)
        perf_record["stage_2_svd_s"] = time.perf_counter() - stage_start
        
        if svd_res is None: continue
        U, S, Vh, U_r, S_inv, V_r, r = svd_res
        
        if hdf5_callback and ("svd" in hdf5_targets or write_all):
            h_start = time.perf_counter()
            hdf5_callback("SVD_Truncation", d_start, d_end, w, s, {
                "U": U.detach().cpu().numpy().astype(np.float32),
                "S": S.detach().cpu().numpy().astype(np.float32),
                "Vh": Vh.detach().cpu().numpy().astype(np.float32),
                "U_r": U_r.detach().cpu().numpy().astype(np.float32),
                "S_inv": S_inv.detach().cpu().numpy().astype(np.float32),
                "V_r": V_r.detach().cpu().numpy().astype(np.float32),
                "r": np.int32(r)
            })
            perf_record["stage_6_hdf5_s"] += time.perf_counter() - h_start

        # --- Stage 3: Operator & Eigen ---
        stage_start = time.perf_counter()
        Atilde, eigvals, W_eig = compute_dmd_operator(U_r, Y, V_r, S_inv)
        perf_record["stage_3_operator_s"] = time.perf_counter() - stage_start
        
        if hdf5_callback:
            h_start = time.perf_counter()
            if "dmd-op" in hdf5_targets or write_all:
                hdf5_callback("Reduced_Operator", d_start, d_end, w, s, {
                    "Atilde": Atilde.detach().cpu().numpy().astype(np.float32)
                })
            
            if "eigen" in hdf5_targets or write_all:
                hdf5_callback("Eigen", d_start, d_end, w, s, {
                    "eigvals_real": eigvals.real.detach().cpu().numpy().astype(np.float32),
                    "eigvals_imag": eigvals.imag.detach().cpu().numpy().astype(np.float32),
                    "W_eig_real": W_eig.real.detach().cpu().numpy().astype(np.float32),
                    "W_eig_imag": W_eig.imag.detach().cpu().numpy().astype(np.float32)
                })
            perf_record["stage_6_hdf5_s"] += time.perf_counter() - h_start

        # --- Stage 4: Modes & Amplitudes ---
        stage_start = time.perf_counter()
        x_last = X[:, -1]
        Phi, b = compute_dmd_modes(Y, V_r, S_inv, W_eig, x_last, dmd_lstsq)
        perf_record["stage_4_modes_s"] = time.perf_counter() - stage_start
        
        if hdf5_callback:
            h_start = time.perf_counter()
            if "dmd-modes" in hdf5_targets or write_all:
                hdf5_callback("DMD_Modes", d_start, d_end, w, s, {
                    "Phi_real": Phi.real.detach().cpu().numpy().astype(np.float32),
                    "Phi_imag": Phi.imag.detach().cpu().numpy().astype(np.float32)
                })
                
            if "dmd_amp" in hdf5_targets or write_all:
                hdf5_callback("DMD_Amplitudes", d_start, d_end, w, s, {
                    "b_real": b.real.detach().cpu().numpy().astype(np.float32),
                    "b_imag": b.imag.detach().cpu().numpy().astype(np.float32)
                })
            perf_record["stage_6_hdf5_s"] += time.perf_counter() - h_start

        # --- Stage 5: Predict ---
        stage_start = time.perf_counter()
        pred_vec_real = reconstruct_and_predict(Phi, eigvals, b)
        perf_record["stage_5_predict_s"] = time.perf_counter() - stage_start
        
        if hdf5_callback and ("pred" in hdf5_targets or write_all):
            h_start = time.perf_counter()
            hdf5_callback("Prediction", d_start, d_end, w, s, {
                "pred_vec_real": pred_vec_real.detach().cpu().numpy().astype(np.float32)
            })
            perf_record["stage_6_hdf5_s"] += time.perf_counter() - h_start

        # --- Stage 7: Tabular Output ---
        stage_start = time.perf_counter()
        record = format_record(pred_vec_real, target_vals, channel_names, d_start, d_end, w, s, rec_type)
        results.append(record)
        
        if record_callback:
            record_callback([record])
            
        perf_record["stage_7_format_s"] = time.perf_counter() - stage_start

        # --- Stack Master Timer ---
        perf_record["total_stack_s"] = time.perf_counter() - sweep_start_time

        perf_records.append(perf_record)

    return results, perf_records


def run_sweeps_gpu_grouped(
    full_data_matrix, dataset_start_idx, dataset_end_idx, verification_vals, enable_train_rec,
    channel_names, stack_range, abort_check=None, record_callback=None, hdf5_callback=None,
    svd_gpu=False, hdf5_targets=None, dmd_lstsq=False, perf_mode=None
):
    """Single-window DMD wrapper for sweeping stack sizes."""
    train_full = torch.tensor(full_data_matrix, device=DEVICE, dtype=torch.float32)
    data_len = train_full.shape[0]

    w = data_len
    if data_len < 2: return [], []

    val_train = train_full[:-1]
    val_targets = train_full[-1, :].cpu().numpy() if enable_train_rec and data_len > 1 else None
    real_targets = verification_vals

    valid_stacks = [s for s in stack_range if s < w and (w - s + 1) >= 2]
    if not valid_stacks: return [], []

    all_results = []
    all_perf_records = []

    if enable_train_rec and val_targets is not None and val_train.shape[0] >= w:
        local_data_val = val_train[-w:, :]
        res, perf = process_window_group(
            local_data_val, "train_rec", val_targets, dataset_start_idx, dataset_end_idx, w, valid_stacks,
            channel_names, abort_check, record_callback, hdf5_callback, svd_gpu, hdf5_targets, dmd_lstsq, perf_mode
        )
        all_results.extend(res)
        all_perf_records.extend(perf)

    local_data_real = train_full[-w:, :]
    res, perf = process_window_group(
        local_data_real, "pred_rec", real_targets, dataset_start_idx, dataset_end_idx, w, valid_stacks,
        channel_names, abort_check, record_callback, hdf5_callback, svd_gpu, hdf5_targets, dmd_lstsq, perf_mode
    )
    all_results.extend(res)
    all_perf_records.extend(perf)
    
    return all_results, all_perf_records

# ==========================================
# BATCHED TENSOR WORKFLOW (HIGH UTILIZATION)
# ==========================================

def run_sweeps_gpu_batched_first(
    full_data_matrix,
    w: int,
    stack_list: list,
    channel_names: list,
    hdf5_callback=None,
    hdf5_targets=None,
    global_start_row=1,
):
    """
    Highly optimized 3D Tensor Batched Workflow (Legacy Functionality).
    """
    data_tensor = torch.tensor(full_data_matrix, device=DEVICE, dtype=torch.float32)
    T, C = data_tensor.shape
    
    if T < w + 1:
        return []

    hdf5_targets = hdf5_targets or []
    write_all = "all" in hdf5_targets
    all_results = []

    for s in stack_list:
        if w - s < 1: continue

        # --- Stage 1: Batched 3D Hankel Construction ---
        windows = data_tensor[:-1].unfold(0, w, 1) 
        B = windows.shape[0]
        num_cols = w - s + 1
        
        h = windows.unfold(2, s, 1).transpose(2, 3) 
        # FIX: Force contiguous memory blocks to prevent cuSOLVER copy-overhead
        H_batch = h.contiguous().view(B, C * s, num_cols)
        X_batch = H_batch[:, :, :-1].contiguous()
        Y_batch = H_batch[:, :, 1:].contiguous()

        # --- Stage 2: Batched SVD ---
        U, S, Vh = torch.linalg.svd(X_batch, full_matrices=False)
        
        max_rank = S.shape[-1]
        r = max(1, int(max_rank * 0.99))
        
        U_r = U[:, :, :r].contiguous()
        S_inv = torch.diag_embed(1.0 / S[:, :r]).contiguous()
        V_r = Vh[:, :r, :].contiguous()

        # --- Stage 3: Batched Operator ---
        Atilde = U_r.transpose(1, 2) @ Y_batch @ V_r.transpose(1, 2) @ S_inv
        eigvals, W_eig = torch.linalg.eig(Atilde)

        # --- Stage 4: Batched Modes & Amplitudes ---
        Y_c = Y_batch.to(torch.complex64)
        V_r_c = V_r.transpose(1, 2).to(torch.complex64)
        S_inv_c = S_inv.to(torch.complex64)
        
        Phi = Y_c @ V_r_c @ S_inv_c @ W_eig
        x_last_c = X_batch[:, :, -1].to(torch.complex64).unsqueeze(-1)
        
        # FIX: Use batched Least-Squares instead of Pseudo-Inverse to prevent CPU fallback
        b = torch.linalg.lstsq(Phi, x_last_c).solution.squeeze(-1)

        # --- Stage 5: Batched Predictions ---
        pred_vec = Phi @ torch.diag_embed(eigvals) @ b.unsqueeze(-1)
        pred_vec_real = pred_vec.squeeze(-1).real
        
        pred_cpu = pred_vec_real.detach().cpu().numpy()

        # --- Stage 6: Batched HDF5 Saving (Optional) ---
        if hdf5_callback:
            batch_d_start = global_start_row - w
            batch_d_end = global_start_row + B - 2 
            
            if "hankle" in hdf5_targets or write_all:
                hdf5_callback("Hankle", batch_d_start, batch_d_end, w, s, {
                    "H_batch": H_batch.detach().cpu().numpy().astype(np.float32),
                    "X_batch": X_batch.detach().cpu().numpy().astype(np.float32),
                    "Y_batch": Y_batch.detach().cpu().numpy().astype(np.float32)
                })
            # ... [Truncated optional HDF5 calls analogous to the main function] ...
            if "pred" in hdf5_targets or write_all:
                hdf5_callback("Prediction", batch_d_start, batch_d_end, w, s, {
                    "pred_vec_real_batch": pred_cpu.astype(np.float32)
                })

        # --- Stage 7: Format CSV Records ---
        for i in range(B):
            curr_pred_row = global_start_row + i
            
            record = {
                "type": "pred_rec",
                "data_set_start": curr_pred_row - w,
                "data_set_end": curr_pred_row - 1,
                "data_window_size": w,
                "window_size": w,
                "stack_size": s,
                "rank_ratio": 0.99,
            }
            
            p_vec = pred_cpu[i]
            target_vals = full_data_matrix[i + w]
            
            for c_idx, ch in enumerate(channel_names):
                idx = (c_idx + 1) * s - 1
                if idx >= len(p_vec): continue
                
                pred_val = float(p_vec[idx])
                record[f"{ch}_pred_value"] = pred_val
                record[f"{ch}_pred_value_int"] = int(round(pred_val))
                
                if target_vals is not None:
                    err_metrics = calculate_prediction_errors(pred_val, float(target_vals[c_idx]))
                    for key, val in err_metrics.items():
                        record[f"{ch}_{key}"] = val
                else:
                    for suffix in ["val_target", "pred_err", "err_pct", "val_target_int", "pred_err_int", "err_pct_int"]:
                        record[f"{ch}_{suffix}"] = None
            
            all_results.append(record)

    return all_results

def run_sweeps_gpu_batched(
    full_data_matrix,
    w: int,
    stack_list: list,
    channel_names: list,
    hdf5_callback=None,
    hdf5_targets=None,
    global_start_row=1,
    perf_mode=None,
    dmd_lstsq=False
):
    """
    Highly optimized 3D Tensor Batched Workflow.
    Slides a fixed window 'w' across the dataset slice and computes all DMDs simultaneously.
    """
    data_tensor = torch.tensor(full_data_matrix, device=DEVICE, dtype=torch.float32)
    T, C = data_tensor.shape
    
    if T < w + 1:
        return [], []

    hdf5_targets = hdf5_targets or []
    write_all = "all" in hdf5_targets
    all_results = []
    perf_records = []

    for s in stack_list:
        if w - s < 1: continue

        sweep_start_time = time.perf_counter()
        perf_record = {"stack_size": s, "window_size": w}
        
        if perf_mode == 'con':
            print(f"\n[Batch Mode] Executing Sweep Set - Stack Size: {s}")

        # --- Stage 1: Batched 3D Hankel Construction ---
        stage_start = time.perf_counter()
        windows = data_tensor[:-1].unfold(0, w, 1) 
        B = windows.shape[0]
        num_cols = w - s + 1
        
        h = windows.unfold(2, s, 1).transpose(2, 3) 
        H_batch = h.contiguous().view(B, C * s, num_cols)
        X_batch = H_batch[:, :, :-1].contiguous()
        Y_batch = H_batch[:, :, 1:].contiguous()
        stage_end = time.perf_counter()
        
        perf_record["stage_1_hankel_s"] = stage_end - stage_start
        if perf_mode == 'con': print(f"  -> Stage 1 (Hankel Construction)  : {perf_record['stage_1_hankel_s']:.5f}s")

        # --- Stage 2: Batched SVD ---
        stage_start = time.perf_counter()
        U, S, Vh = torch.linalg.svd(X_batch, full_matrices=False)
        
        max_rank = S.shape[-1]
        r = max(1, int(max_rank * 0.99))
        
        U_r = U[:, :, :r].contiguous()
        S_inv = torch.diag_embed(1.0 / S[:, :r]).contiguous()
        V_r = Vh[:, :r, :].contiguous()
        stage_end = time.perf_counter()
        
        perf_record["stage_2_svd_s"] = stage_end - stage_start
        if perf_mode == 'con': print(f"  -> Stage 2 (SVD Truncation)       : {perf_record['stage_2_svd_s']:.5f}s")

        # --- Stage 3: Batched Operator ---
        stage_start = time.perf_counter()
        Atilde = U_r.transpose(1, 2) @ Y_batch @ V_r.transpose(1, 2) @ S_inv
        eigvals, W_eig = torch.linalg.eig(Atilde)
        stage_end = time.perf_counter()
        
        perf_record["stage_3_operator_s"] = stage_end - stage_start
        if perf_mode == 'con': print(f"  -> Stage 3 (Operator & Eigen)     : {perf_record['stage_3_operator_s']:.5f}s")

        # --- Stage 4: Batched Modes & Amplitudes ---
        stage_start = time.perf_counter()
        Y_c = Y_batch.to(torch.complex64)
        V_r_c = V_r.transpose(1, 2).to(torch.complex64)
        S_inv_c = S_inv.to(torch.complex64)
        
        Phi = Y_c @ V_r_c @ S_inv_c @ W_eig
        x_last_c = X_batch[:, :, -1].to(torch.complex64).unsqueeze(-1)
        
        if dmd_lstsq:
            b = torch.linalg.lstsq(Phi, x_last_c).solution.squeeze(-1)
        else:
            b = (torch.linalg.pinv(Phi) @ x_last_c).squeeze(-1)
            
        stage_end = time.perf_counter()
        
        perf_record["stage_4_modes_s"] = stage_end - stage_start
        if perf_mode == 'con': print(f"  -> Stage 4 (Modes & Amplitudes)   : {perf_record['stage_4_modes_s']:.5f}s")

        # --- Stage 5: Batched Predictions ---
        stage_start = time.perf_counter()
        pred_vec = Phi @ torch.diag_embed(eigvals) @ b.unsqueeze(-1)
        pred_vec_real = pred_vec.squeeze(-1).real
        
        pred_cpu = pred_vec_real.detach().cpu().numpy()
        stage_end = time.perf_counter()
        
        perf_record["stage_5_predict_s"] = stage_end - stage_start
        if perf_mode == 'con': print(f"  -> Stage 5 (Predictions)          : {perf_record['stage_5_predict_s']:.5f}s")

        # --- Stage 6: Batched HDF5 Saving (Optional) ---
        stage_start = time.perf_counter()
        if hdf5_callback:
            batch_d_start = global_start_row - w
            batch_d_end = global_start_row + B - 2 
            
            if "hankle" in hdf5_targets or write_all:
                hdf5_callback("Hankle", batch_d_start, batch_d_end, w, s, {
                    "H_batch": H_batch.detach().cpu().numpy().astype(np.float32),
                    "X_batch": X_batch.detach().cpu().numpy().astype(np.float32),
                    "Y_batch": Y_batch.detach().cpu().numpy().astype(np.float32)
                })
            if "svd" in hdf5_targets or write_all:
                hdf5_callback("SVD_Truncation", batch_d_start, batch_d_end, w, s, {
                    "U_batch": U.detach().cpu().numpy().astype(np.float32),
                    "S_batch": S.detach().cpu().numpy().astype(np.float32),
                    "Vh_batch": Vh.detach().cpu().numpy().astype(np.float32),
                    "U_r_batch": U_r.detach().cpu().numpy().astype(np.float32),
                    "S_inv_batch": S_inv.detach().cpu().numpy().astype(np.float32),
                    "V_r_batch": V_r.detach().cpu().numpy().astype(np.float32),
                    "r_batch": np.int32(r)
                })
            if "dmd-op" in hdf5_targets or write_all:
                hdf5_callback("Reduced_Operator", batch_d_start, batch_d_end, w, s, {
                    "Atilde_batch": Atilde.detach().cpu().numpy().astype(np.float32)
                })
            if "eigen" in hdf5_targets or write_all:
                hdf5_callback("Eigen", batch_d_start, batch_d_end, w, s, {
                    "eigvals_real_batch": eigvals.real.detach().cpu().numpy().astype(np.float32),
                    "eigvals_imag_batch": eigvals.imag.detach().cpu().numpy().astype(np.float32),
                    "W_eig_real_batch": W_eig.real.detach().cpu().numpy().astype(np.float32),
                    "W_eig_imag_batch": W_eig.imag.detach().cpu().numpy().astype(np.float32)
                })
            if "dmd-modes" in hdf5_targets or write_all:
                hdf5_callback("DMD_Modes", batch_d_start, batch_d_end, w, s, {
                    "Phi_real_batch": Phi.real.detach().cpu().numpy().astype(np.float32),
                    "Phi_imag_batch": Phi.imag.detach().cpu().numpy().astype(np.float32)
                })
            if "dmd_amp" in hdf5_targets or write_all:
                hdf5_callback("DMD_Amplitudes", batch_d_start, batch_d_end, w, s, {
                    "b_real_batch": b.real.detach().cpu().numpy().astype(np.float32),
                    "b_imag_batch": b.imag.detach().cpu().numpy().astype(np.float32)
                })
            if "pred" in hdf5_targets or write_all:
                hdf5_callback("Prediction", batch_d_start, batch_d_end, w, s, {
                    "pred_vec_real_batch": pred_cpu.astype(np.float32)
                })
        stage_end = time.perf_counter()
        
        perf_record["stage_6_hdf5_s"] = stage_end - stage_start
        if perf_mode == 'con': print(f"  -> Stage 6 (HDF5 I/O Saving)      : {perf_record['stage_6_hdf5_s']:.5f}s")

        # --- Stage 7: Format CSV Records ---
        stage_start = time.perf_counter()
        for i in range(B):
            curr_pred_row = global_start_row + i
            
            record = {
                "type": "pred_rec",
                "data_set_start": curr_pred_row - w,
                "data_set_end": curr_pred_row - 1,
                "data_window_size": w,
                "window_size": w,
                "stack_size": s,
                "rank_ratio": 0.99,
            }
            
            p_vec = pred_cpu[i]
            target_vals = full_data_matrix[i + w]
            
            for c_idx, ch in enumerate(channel_names):
                idx = (c_idx + 1) * s - 1
                if idx >= len(p_vec): continue
                
                pred_val = float(p_vec[idx])
                record[f"{ch}_pred_value"] = pred_val
                record[f"{ch}_pred_value_int"] = int(round(pred_val))
                
                if target_vals is not None:
                    err_metrics = calculate_prediction_errors(pred_val, float(target_vals[c_idx]))
                    for key, val in err_metrics.items():
                        record[f"{ch}_{key}"] = val
                else:
                    for suffix in ["val_target", "pred_err", "err_pct", "val_target_int", "pred_err_int", "err_pct_int"]:
                        record[f"{ch}_{suffix}"] = None
            
            all_results.append(record)
        stage_end = time.perf_counter()
        
        perf_record["stage_7_format_s"] = stage_end - stage_start
        if perf_mode == 'con': print(f"  -> Stage 7 (Format Tabular Recs)  : {perf_record['stage_7_format_s']:.5f}s")
        
        # --- End of Sweep Master Timer ---
        sweep_end_time = time.perf_counter()
        perf_record["total_sweep_s"] = sweep_end_time - sweep_start_time
        
        if perf_mode == 'con':
            print(f"  ================================================")
            print(f"  -> Total Sweep Time               : {perf_record['total_sweep_s']:.5f}s")
        else:
            # Clean standard progress indicator for batched mode
            end_pred_row = global_start_row + B - 1
            print(f"Processed Stack {s} (Rows: {global_start_row}-{end_pred_row}, Window: {w}) ({perf_record['total_sweep_s']:.2f}s)")
            
        perf_records.append(perf_record)

    return all_results, perf_records