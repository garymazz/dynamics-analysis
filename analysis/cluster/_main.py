import time
import torch
import numpy as np
from analysis.dmd._main import process_window_group, run_sweeps_gpu_batched, DEVICE
from meta_analysis.evaluator import calculate_prediction_errors # For base metrics
from meta_analysis.ensemble import calculate_ensemble_statistics # For median stats

def filter_optimal_configurations(results: list, primary_ch: str, error_threshold: float) -> list:
    """
    Independent Quantitative Filtering: 
    Gatekeeper function to ensure model fidelity before ensembling.
    """
    valid_configs = []
    for res in results:
        err = res.get(f"{primary_ch}_err_pct")
        if err is not None and err <= error_threshold:
            valid_configs.append(res)
    return valid_configs

# INDEPENDENT QUALITATIVE & QUANTITATIVE ANALYSIS
# ==========================================

def filter_optimal_configurations(results: list, primary_ch: str, error_threshold: float) -> list:
    """
    Qualitative analysis: Filters configurations meeting the error threshold heuristic.
    """
    valid_configs = []
    for res in results:
        err = res.get(f"{primary_ch}_err_pct")
        if err is not None and err <= error_threshold:
            valid_configs.append(res)
    return valid_configs

def filter_optimal_configurations_batched(batch_results: list, valid_stacks: list, primary_ch: str, error_threshold: float, w: int) -> list:
    """
    Qualitative & Quantitative hybrid: Computes mean batch error and filters valid configs.
    """
    best_configs = []
    stack_errors = {s: [] for s in valid_stacks}
    
    for res in batch_results:
        s = res["stack_size"]
        err = res.get(f"{primary_ch}_err_pct")
        if err is not None:
            stack_errors[s].append(err)
            
    for s, errors in stack_errors.items():
        if not errors: continue
        mean_err = sum(errors) / len(errors) # Quantitative Mean
        if mean_err <= error_threshold:      # Qualitative Filter
            best_configs.append({
                "window_size": w,
                "stack_size": s,
                f"{primary_ch}_err_pct": mean_err
            })
    return best_configs

def calculate_ensemble_statistics(preds: list, target_val: float = None) -> dict:
    """
    Quantitative analysis: Computes statistical median and uncertainty bounds.
    """
    if not preds: return {}
    
    med_pred = float(np.median(preds))
    std_pred = float(np.std(preds))
    stats = {
        "prediction_median": med_pred,
        "uncertainty_std": std_pred
    }
    
    if target_val is not None:
        truth = float(target_val)
        stats["actual"] = truth
        stats["error"] = abs(med_pred - truth)
        
    return stats


# ==========================================
# CLUSTER WORKFLOW ORCHESTRATORS
# ==========================================

def run_cluster_forecast_workflow(
    full_data_matrix: np.ndarray,
    channel_names: list,
    error_threshold: float,
    min_window: int,
    max_window: int,
    min_stack: int,
    max_stack: int,
    ensemble_width: int,
    forecast_row: int = None,
    detected_period: int = None,
    abort_check=None,
    svd_gpu: bool = False,
    log_callback=None,
    perf_mode=None
) -> tuple:
    """
    Executes the Smart Mode Cluster-DMD Forecasting Workflow (Sequential Optimization).
    """
    def log(msg):
        if log_callback: log_callback(msg)
        else: print(msg)

    total_len = len(full_data_matrix)
    global_perf_data = []

    # --- 1. Target Resolution ---
    if forecast_row is not None:
        target_row = forecast_row
        log(f"Target Mode: User Defined Row {target_row}")
    else:
        target_row = total_len + 1
        log(f"Target Mode: Future Forecast (Row {target_row})")

    target_idx = target_row - 1
    if target_idx > total_len:
        log("[Error] Target row is too far in the future.")
        return None, []

    target_vals = None
    if target_idx < total_len:
        target_vals = full_data_matrix[target_idx, :]
        log("Ground Truth: Available")
    else:
        log("Ground Truth: Not Available")

    train_data = full_data_matrix[:target_idx]

    # --- 2. Optimization Phase ---
    log(f"\nStep 1: Optimization (Scanning recent history for needles < {error_threshold}% error)...")
    opt_start_time = time.perf_counter()

    opt_target_idx = target_idx - 1
    if opt_target_idx < max_window:
        log(f"[Warning] History ({opt_target_idx}) is shorter than max window ({max_window}). Optimization may be limited.")

    opt_train_data = full_data_matrix[:opt_target_idx]
    opt_truth = full_data_matrix[opt_target_idx, :]

    if detected_period:
        log(f"Optimizing Search Space using Detected Period T={detected_period}...")
        pilot_windows = []
        multiplier = 1
        while True:
            center = detected_period * multiplier
            if center > max_window: break
            if center >= min_window:
                pilot_windows.extend(range(center - 2, center + 3))
            multiplier += 1
        pilot_windows = sorted(list(set(pilot_windows)))
    else:
        log(f"No period detected. Using Full Spectrum Scan ({min_window}-{max_window}).")
        pilot_windows = range(min_window, max_window, 1)

    pilot_stacks = range(min_stack, max_stack, 2)
    best_configs = []
    train_tensor = torch.tensor(opt_train_data, device=DEVICE, dtype=torch.float32)
    data_len = train_tensor.shape[0]
    primary_ch = channel_names[0]

    for w in pilot_windows:
        if abort_check and abort_check(): return None, []
        if data_len < w: continue
            
        valid_stacks = [s for s in pilot_stacks if s < w and (w - s + 1) >= 2]
        if not valid_stacks: continue

        local_data = train_tensor[-w:, :]
        
        results, perf_records = process_window_group(
            local_data=local_data, rec_type="opt", target_vals=opt_truth,
            d_start=0, d_end=opt_target_idx, w=w, stack_list=valid_stacks,
            channel_names=channel_names, abort_check=abort_check, svd_gpu=svd_gpu, perf_mode=perf_mode
        )
        
        if perf_records:
            for p in perf_records: p["cluster_phase"] = "optimization"
            global_perf_data.extend(perf_records)

        # ---> DEPLOY INDEPENDENT QUALITATIVE ANALYSIS <---
        valid_configs = filter_optimal_configurations(results, primary_ch, error_threshold)
        best_configs.extend(valid_configs)

    opt_end_time = time.perf_counter()

    if not best_configs:
        log(f"[Warning] No configurations found with error < {error_threshold}%.")
        return None, global_perf_data

    # Retrieve the absolute best configuration
    best_configs.sort(key=lambda x: x[f"{primary_ch}_err_pct"])
    top_config = best_configs[0]

    w_opt = top_config["window_size"]
    s_opt = top_config["stack_size"]
    min_err = top_config[f"{primary_ch}_err_pct"]

    log(f"[Success] Optimal Config Found: Window={w_opt}, Stack={s_opt} (Prev Error: {min_err:.4f}%) in {opt_end_time - opt_start_time:.2f}s")

    # --- 3. Ensemble Forecast Phase ---
    log(f"\nStep 2: Ensemble Forecast (Width +/- {ensemble_width})....")
    ens_start_time = time.perf_counter()
    
    ensemble_windows = range(w_opt - ensemble_width, w_opt + ensemble_width + 1)
    ensemble_stack = [s_opt]
    train_tensor_final = torch.tensor(train_data, device=DEVICE, dtype=torch.float32)
    ensemble_results = []

    for w in ensemble_windows:
        if abort_check and abort_check(): return None, []
        if w < min_window or train_tensor_final.shape[0] < w: continue
            
        local_data = train_tensor_final[-w:, :]
        
        results, perf_records = process_window_group(
            local_data=local_data, rec_type="forecast", target_vals=target_vals,
            d_start=0, d_end=target_idx, w=w, stack_list=ensemble_stack,
            channel_names=channel_names, abort_check=abort_check, svd_gpu=svd_gpu, perf_mode=perf_mode
        )
        
        if perf_records:
            for p in perf_records: p["cluster_phase"] = "ensemble"
            global_perf_data.extend(perf_records)
            
        ensemble_results.extend(results)

    ens_end_time = time.perf_counter()

    # --- 4. Statistical Aggregation ---
    log(f"\n=== Forecast Results (Generated in {ens_end_time - ens_start_time:.2f}s) ===")
    final_payload = {
        "target_row": target_row, "optimal_window_center": w_opt, "optimal_stack": s_opt,
        "period_used": detected_period, "ensemble_count": len(ensemble_results), "channels": {}
    }

    # ---> DEPLOY INDEPENDENT QUANTITATIVE ANALYSIS <---
    for idx, ch in enumerate(channel_names):
        preds = [r.get(f"{ch}_pred_value") for r in ensemble_results if f"{ch}_pred_value" in r]
        t_val = float(target_vals[idx]) if target_vals is not None else None
        
        ch_data = calculate_ensemble_statistics(preds, target_val=t_val)
        
        if ch_data:
            log(f"Channel {ch}: {ch_data['prediction_median']:.4f} (+/- {ch_data['uncertainty_std']:.4f})")
            if 'actual' in ch_data: log(f"   Actual: {ch_data['actual']:.4f}, Error: {ch_data['error']:.4f}")
            final_payload["channels"][ch] = ch_data

    return final_payload, global_perf_data


def run_cluster_forecast_workflow_batched(
    full_data_matrix: np.ndarray,
    channel_names: list,
    error_threshold: float,
    min_window: int,
    max_window: int,
    min_stack: int,
    max_stack: int,
    ensemble_width: int,
    forecast_row: int = None,
    detected_period: int = None,
    abort_check=None,
    svd_gpu: bool = False,
    log_callback=None,
    perf_mode=None,
    validation_batch_size: int = 20
) -> tuple:
    """
    Executes the Smart Mode Cluster-DMD Forecasting Workflow utilizing the Batched GPU Engine.
    Optimizes hyperparameter selection by evaluating a block of historical data simultaneously.
    """
    def log(msg):
        if log_callback: log_callback(msg)
        else: print(msg)

    total_len = len(full_data_matrix)
    global_perf_data = []

    # --- 1. Target Resolution ---
    if forecast_row is not None:
        target_row = forecast_row
        log(f"Target Mode: User Defined Row {target_row}")
    else:
        target_row = total_len + 1
        log(f"Target Mode: Future Forecast (Row {target_row})")

    target_idx = target_row - 1
    if target_idx > total_len:
        log("[Error] Target row is too far in the future.")
        return None, []

    target_vals = None
    if target_idx < total_len:
        target_vals = full_data_matrix[target_idx, :]
        log("Ground Truth: Available")
    else:
        log("Ground Truth: Not Available")

    train_data = full_data_matrix[:target_idx]

    # --- 2. Optimization Phase (Batched Block Evaluation) ---
    log(f"\nStep 1: Batched Optimization (Evaluating last {validation_batch_size} rows for mean error < {error_threshold}%)...")
    opt_start_time = time.perf_counter()

    batch_end_idx = target_idx - 1
    batch_start_idx = batch_end_idx - validation_batch_size + 1
    
    if batch_start_idx < max_window:
        log(f"[Warning] History is shorter than max window. Validation batch may be truncated.")
        batch_start_idx = max_window

    if detected_period:
        log(f"Optimizing Search Space using Detected Period T={detected_period}...")
        pilot_windows = []
        multiplier = 1
        while True:
            center = detected_period * multiplier
            if center > max_window: break
            if center >= min_window:
                pilot_windows.extend(range(center - 2, center + 3))
            multiplier += 1
        pilot_windows = sorted(list(set(pilot_windows)))
    else:
        log(f"No period detected. Using Full Spectrum Scan ({min_window}-{max_window}).")
        pilot_windows = range(min_window, max_window, 1)

    pilot_stacks = range(min_stack, max_stack, 2)
    best_configs = []
    primary_ch = channel_names[0]

    for w in pilot_windows:
        if abort_check and abort_check(): return None, []
        
        slice_start = batch_start_idx - w
        if slice_start < 0: continue
            
        valid_stacks = [s for s in pilot_stacks if s < w and (w - s + 1) >= 2]
        if not valid_stacks: continue

        # Extract the entire block needed for batched predictions
        data_slice = full_data_matrix[slice_start : batch_end_idx + 1]
        
        # Dispatch to massively parallel batched engine
        batch_results, perf_records = run_sweeps_gpu_batched(
            full_data_matrix=data_slice,
            w=w,
            stack_list=valid_stacks,
            channel_names=channel_names,
            global_start_row=batch_start_idx,
            perf_mode=perf_mode
        )
        
        if perf_records:
            for p in perf_records: p["cluster_phase"] = "optimization_batched"
            global_perf_data.extend(perf_records)

        # ---> DEPLOY INDEPENDENT QUALITATIVE ANALYSIS (HYBRID MEAN BATCH FILTER) <---
        valid_batch_configs = filter_optimal_configurations_batched(
            batch_results=batch_results, 
            valid_stacks=valid_stacks, 
            primary_ch=primary_ch, 
            error_threshold=error_threshold, 
            w=w
        )
        best_configs.extend(valid_batch_configs)

    opt_end_time = time.perf_counter()

    if not best_configs:
        log(f"[Warning] No batched configurations found with mean error < {error_threshold}%.")
        return None, global_perf_data

    # Retrieve the absolute best configuration
    best_configs.sort(key=lambda x: x[f"{primary_ch}_err_pct"])
    top_config = best_configs[0]

    w_opt = top_config["window_size"]
    s_opt = top_config["stack_size"]
    min_err = top_config[f"{primary_ch}_err_pct"]

    log(f"[Success] Optimal Batched Config Found: Window={w_opt}, Stack={s_opt} (Mean Prev Error: {min_err:.4f}%) in {opt_end_time - opt_start_time:.2f}s")

    # --- 3. Ensemble Forecast Phase (Sequential) ---
    log(f"\nStep 2: Ensemble Forecast (Width +/- {ensemble_width})....")
    ens_start_time = time.perf_counter()
    
    ensemble_windows = range(w_opt - ensemble_width, w_opt + ensemble_width + 1)
    ensemble_stack = [s_opt]
    train_tensor_final = torch.tensor(train_data, device=DEVICE, dtype=torch.float32)
    ensemble_results = []

    # Ensemble relies on varying window sizes, so it remains sequential
    for w in ensemble_windows:
        if abort_check and abort_check(): return None, []
        if w < min_window or train_tensor_final.shape[0] < w: continue
            
        local_data = train_tensor_final[-w:, :]
        
        results, perf_records = process_window_group(
            local_data=local_data, rec_type="forecast", target_vals=target_vals,
            d_start=0, d_end=target_idx, w=w, stack_list=ensemble_stack,
            channel_names=channel_names, abort_check=abort_check, svd_gpu=svd_gpu, perf_mode=perf_mode
        )
        
        if perf_records:
            for p in perf_records: p["cluster_phase"] = "ensemble"
            global_perf_data.extend(perf_records)
            
        ensemble_results.extend(results)

    ens_end_time = time.perf_counter()

    # --- 4. Statistical Aggregation ---
    log(f"\n=== Forecast Results (Generated in {ens_end_time - ens_start_time:.2f}s) ===")
    final_payload = {
        "target_row": target_row, "optimal_window_center": w_opt, "optimal_stack": s_opt,
        "period_used": detected_period, "ensemble_count": len(ensemble_results), "channels": {}
    }

    # ---> DEPLOY INDEPENDENT QUANTITATIVE ANALYSIS <---
    for idx, ch in enumerate(channel_names):
        preds = [r.get(f"{ch}_pred_value") for r in ensemble_results if f"{ch}_pred_value" in r]
        t_val = float(target_vals[idx]) if target_vals is not None else None
        
        ch_data = calculate_ensemble_statistics(preds, target_val=t_val)
        
        if ch_data:
            log(f"Channel {ch}: {ch_data['prediction_median']:.4f} (+/- {ch_data['uncertainty_std']:.4f})")
            if 'actual' in ch_data: log(f"   Actual: {ch_data['actual']:.4f}, Error: {ch_data['error']:.4f}")
            final_payload["channels"][ch] = ch_data

    return final_payload, global_perf_data