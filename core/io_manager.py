import os
import json
import pandas as pd

def get_schema_definition(channel_names):
    """Generates the JSON schema definition for the flat CSV/Parquet records."""
    fields = [
        {"name": "type", "type": "string"},
        {"name": "data_set_start", "type": "integer"},
        {"name": "data_set_end", "type": "integer"},
        {"name": "data_window_size", "type": "integer"},
        {"name": "window_size", "type": "integer"},
        {"name": "stack_size", "type": "integer"},
        {"name": "rank_ratio", "type": "float"},
    ]
    for ch in channel_names:
        fields.extend([
            {"name": f"{ch}_val_target", "type": "float"},
            {"name": f"{ch}_pred_value", "type": "float"},
            {"name": f"{ch}_pred_err", "type": "float"},
            {"name": f"{ch}_err_pct", "type": "float"},
            {"name": f"{ch}_val_target_int", "type": "integer"},
            {"name": f"{ch}_pred_value_int", "type": "integer"},
            {"name": f"{ch}_pred_err_int", "type": "integer"},
            {"name": f"{ch}_err_pct_int", "type": "float"},
        ])
    return {"fields": fields}


class ConfigManager:
    """Handles saving and loading run configurations for the Resume workflow."""
    @staticmethod
    def save_run_config(args, output_base):
        config_file = f"{output_base}_config.json"
        try:
            with open(config_file, "w") as f:
                json.dump(vars(args), f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save config file: {e}")

    @staticmethod
    def load_run_config_dict(output_base):
        config_file = f"{output_base}_config.json"
        if os.path.exists(config_file):
            print(f"Resuming: Loading configuration from {config_file}")
            try:
                with open(config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Config file corrupt, cannot resume ({e}).")
                return None
        print("Resuming: No config file found.")
        return None

    @staticmethod
    def compute_ignored_cli_options(cli_args, config_dict):
        ignored = []
        name_to_flag = {
            "input": "--input", "output": "--output", "hdf5_dir": "--hdf5-dir",
            "format": "--format", "keep_temp": "--keep-temp", "channels": "--channels",
            "start_row": "--start-row", "end_row": "--end-row", "dec_end": "--dec-end",
            "inc_start": "--inc-start", "train_rec": "--train-rec", "schema": "--schema",
            "min_stack": "--min-stack", "max_stack": "--max-stack",
            "min_window": "--min-window", "max_window": "--max-window",
            "svd_gpu": "--svd-gpu", "hdf5": "--hdf5",
        }

        for attr, flag in name_to_flag.items():
            if attr == "resume" or not hasattr(cli_args, attr):
                continue
            cli_val = getattr(cli_args, attr)
            cfg_val = config_dict.get(attr)
            
            if isinstance(cli_val, (list, tuple)) and isinstance(cfg_val, (list, tuple)):
                equal = list(cli_val) == list(cfg_val)
            else:
                equal = cli_val == cfg_val
                
            if not equal:
                ignored.append(flag)
        return sorted(set(ignored))

    @staticmethod
    def apply_config_to_args(config_dict, args):
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
        if hasattr(args, "resume"):
            args.resume = True
        return args


class IOManager:
    """
    Handles all Pandas data loading, temporary CSV buffering, and final file conversion.
    Encapsulates state to completely eliminate global variables.
    """
    def __init__(self, output_base, buffer_size=1):
        self.output_base = output_base
        self.buffer_size = buffer_size
        self.results_buffer = []
        self.temp_file = f"{self.output_base}_temp.csv"

    @staticmethod
    def load_data(file_path, channel_names):
        """Loads data, maps channels."""
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_excel(file_path, header=None)

        num_requested = len(channel_names)
        if df.shape[1] < num_requested:
            raise ValueError(f"Input file has {df.shape[1]} columns, but {num_requested} channels requested.")

        selected_indices = []
        for ch in channel_names:
            try:
                idx = int(ch.replace("S", "")) - 1
                selected_indices.append(idx)
            except Exception:
                raise ValueError(f"Invalid channel name format: {ch}. Expected S1, S2, etc.")

        df_selected = df.iloc[:, selected_indices]
        df_selected.columns = channel_names
        return df_selected

    def record_callback(self, results):
        self.results_buffer.extend(results)
        if len(self.results_buffer) >= self.buffer_size:
            self.flush_buffer()

    def flush_buffer(self):
        if not self.results_buffer:
            return
        df = pd.DataFrame(self.results_buffer)
        header = not os.path.exists(self.temp_file)
        try:
            df.to_csv(self.temp_file, mode="a", header=header, index=False)
        except Exception as e:
            print(f"Error flushing buffer to {self.temp_file}: {e}")
        self.results_buffer = []

    def _count_csv_fields(self, line: str) -> int:
        in_quote = False
        count = 1 if line and line.strip() else 0
        for ch in line:
            if ch == '"':
                in_quote = not in_quote
            elif ch == "," and not in_quote:
                count += 1
        return count

    def verify_and_fix_temp_file(self):
        if not os.path.exists(self.temp_file):
            return None
        try:
            df = pd.read_csv(self.temp_file)
            if df.empty:
                return None
            last_row = df.iloc[-1]
            if pd.isna(last_row.get("window_size")) or pd.isna(last_row.get("stack_size")):
                print("Resuming: Last row of temp file is incomplete. Removing it.")
                df_fixed = df.iloc[:-1]
                df_fixed.to_csv(self.temp_file, index=False)
                return df_fixed
            return df
        except Exception as e:
            print(f"Warning: Temp file unreadable ({e}). Starting fresh.")
            return None

    def determine_resume_state_granular(self, mode_inc_start):
        """Analyzes temp/output files to determine where a killed run left off."""
        df_all = None
        if os.path.exists(self.temp_file):
            df_all = self.verify_and_fix_temp_file()

        if df_all is None or df_all.empty:
            for fname, ftype in [(f"{self.output_base}.parquet", "parquet"), (f"{self.output_base}.csv", "csv")]:
                if os.path.exists(fname):
                    try:
                        df_all = pd.read_parquet(fname) if ftype == "parquet" else pd.read_csv(fname)
                        if not df_all.empty:
                            break
                    except Exception:
                        pass

        if df_all is None or df_all.empty:
            return None

        col_row = "data_set_start" if mode_inc_start else "data_set_end"
        return {
            "row": int(df_all[col_row].iloc[-1]),
            "window": int(df_all["window_size"].iloc[-1]),
            "stack": int(df_all["stack_size"].iloc[-1]),
        }

    def truncate_corrupt_rows(self) -> None:
        if not os.path.exists(self.temp_file):
            return
        with open(self.temp_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return
        header = lines[0]
        expected_fields = self._count_csv_fields(header)
        last_good_idx = 0 
        for idx in range(1, len(lines)):
            line = lines[idx]
            if not line.strip():
                continue
            n_fields = self._count_csv_fields(line)
            if n_fields == expected_fields:
                last_good_idx = idx
            else:
                print(f"Resuming: Detected corrupt CSV row {idx + 1}. Truncating.")
                break
        else:
            return
        good_lines = lines[: last_good_idx + 1]
        with open(self.temp_file, "w", encoding="utf-8") as f:
            f.writelines(good_lines)

    def cleanup_and_merge(self, keep_temp=False, format_type="both"):
        self.flush_buffer()
        if not os.path.exists(self.temp_file):
            print("\nNo results to save.")
            return

        self.truncate_corrupt_rows()
        print(f"\nFinalizing output from {self.temp_file}...")

        try:
            df_final = pd.read_csv(self.temp_file)
            if format_type in ["parquet", "both"]:
                out_name = f"{self.output_base}.parquet"
                df_final.to_parquet(out_name, index=False)
                print(f"Saved: {out_name}")
            if format_type in ["xlsx", "both"]:
                out_name = f"{self.output_base}.xlsx"
                if len(df_final) > 1000000:
                    print("Warning: Data exceeds Excel row limit. Skipping .xlsx save.")
                else:
                    df_final.to_excel(out_name, index=False)
                    print(f"Saved: {out_name}")

            if not keep_temp:
                os.remove(self.temp_file)
            else:
                print(f"Temp file kept: {self.temp_file}")
        except Exception as e:
            print(f"Error during finalization: {e}")