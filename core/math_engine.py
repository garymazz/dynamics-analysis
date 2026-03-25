
import torch
import numpy as np

# Device Configuration evaluated at module load
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    Core PyTorch DMD/SVD calculation engine.
    Framework-agnostic: relies on callbacks for interruption (abort_check) and data saving.
    """
    num_channels = len(channel_names)
    results = []

    for s in stack_list:
        # 1. Framework-Agnostic Graceful Exit Check
        if abort_check and abort_check():
            break
            
        channels_unfolded = []
        try:
            for c in range(num_channels):
                if local_data.shape[0] < w:
                    continue
                channels_unfolded.append(local_data[:, c].unfold(0, s, 1).T)
        except Exception:
            continue

        if not channels_unfolded:
            continue

        H = torch.cat(channels_unfolded, dim=0)
        X = H[:, :-1]
        Y = H[:, 1:]

        # Choose device for SVD
        if svd_gpu and DEVICE == "cuda":
            X_svd = X.to("cuda")
        else:
            X_svd = X

        try:
            U_svd, S_svd, Vh_svd = torch.linalg.svd(X_svd, full_matrices=False)
        except Exception:
            continue

        # Bring results back to the working DEVICE
        if X_svd.device.type == "cuda" and DEVICE == "cpu":
            U = U_svd.to("cpu")
            S = S_svd.to("cpu")
            Vh = Vh_svd.to("cpu")
        else:
            U = U_svd.to(DEVICE)
            S = S_svd.to(DEVICE)
            Vh = Vh_svd.to(DEVICE)

        max_rank = len(S)
        r = max(1, int(max_rank * 0.99))
        U_r = U[:, :r]
        S_inv = torch.diag(1.0 / S[:r])
        V_r = Vh[:r, :]

        Atilde = U_r.T @ Y @ V_r.T @ S_inv
        eigvals, W_eig = torch.linalg.eig(Atilde)

        Y_c = Y.to(torch.complex64)
        V_r_c = V_r.T.to(torch.complex64)
        S_inv_c = S_inv.to(torch.complex64)

        Phi = Y_c @ V_r_c @ S_inv_c @ W_eig
        x_last = X[:, -1].to(torch.complex64)
        b = torch.linalg.pinv(Phi) @ x_last

        pred_vec = Phi @ torch.diag(eigvals) @ b
        pred_vec_real = pred_vec.real

        # 2. Framework-Agnostic HDF5 Injection
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

        # 3. Compile Standard Flat Record
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

        results.append(record)
        
        # 4. Framework-Agnostic Buffer Streaming
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