
import os
import pandas as pd

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
    def load_data(file_path, channel_names, time_col=None, start_time=None, end_time=None):
        """Loads data, maps channels, and optionally filters by TimeIntervals."""
        import pandas as pd
        
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_excel(file_path, header=None)

        # Time Interval Filtering
        if time_col is not None:
            try:
                # Assuming time_col is the column name or index
                col_idx = int(time_col) if str(time_col).isdigit() else time_col
                df['__parsed_time'] = pd.to_datetime(df.iloc[:, col_idx] if isinstance(col_idx, int) else df[col_idx])
                
                if start_time:
                    df = df[df['__parsed_time'] >= pd.to_datetime(start_time)]
                if end_time:
                    df = df[df['__parsed_time'] <= pd.to_datetime(end_time)]
                    
                df = df.drop(columns=['__parsed_time']).reset_index(drop=True)
                print(f"[Info] TimeInterval filter applied. Rows remaining: {len(df)}")
            except Exception as e:
                print(f"[Warning] Failed to parse TimeIntervals: {e}")

        # Channel Mapping
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
        """Callback to receive data from the math engine and buffer it."""
        self.results_buffer.extend(results)
        if len(self.results_buffer) >= self.buffer_size:
            self.flush_buffer()

    def flush_buffer(self):
        """Appends the current buffer to the temporary CSV and clears it."""
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

    def truncate_corrupt_rows(self) -> None:
        """Safely truncates the temporary CSV if a system kill interrupted a write."""
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
                print(
                    f"Resuming: Detected corrupt CSV row {idx + 1} "
                    f"(expected {expected_fields} fields, saw {n_fields}). "
                    f"Truncating file after row {last_good_idx + 1}."
                )
                break
        else:
            return

        good_lines = lines[: last_good_idx + 1]
        with open(self.temp_file, "w", encoding="utf-8") as f:
            f.writelines(good_lines)

    def verify_and_fix_temp_file(self):
        """Verifies the CSV integrity for the resume state logic."""
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

    def cleanup_and_merge(self, keep_temp=False, format_type="both"):
        """Finalizes the workflow by converting the temp CSV into Parquet/Excel."""
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
                print(f"Temporary file {self.temp_file} removed.")
            else:
                print(f"Temp file kept: {self.temp_file}")

        except Exception as e:
            print(f"Error during finalization: {e}")