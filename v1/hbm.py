import numpy as np
import pandas as pd
import argparse
import sys
import os
from tqdm import tqdm 
import warnings
import multiprocessing
from typing import Tuple, List, Dict, Any, Optional

# Dependencies for Holt-Winters:
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

# Suppress warnings for clean console output
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Core Estimation Class (Stabilized Holt-Winters V2) ---

class HoltWintersEstimator:
    """
    Stabilized V2 Estimator using Holt-Winters (Triple Exponential Smoothing)
    to model level, trend, and simple confidence intervals, avoiding complex dependencies.
    """
    def __init__(self, data_3d, entity_names, time_index, field_cols):
        self.data_3d = data_3d
        self.entities = entity_names
        self.fields = field_cols
        self.n_entities, self.n_timesteps, self.n_fields = data_3d.shape
        
        # Initialize results arrays
        self.smoothed_data = np.full_like(data_3d, np.nan)
        self.hdi_3 = np.full_like(data_3d, np.nan)
        self.hdi_97 = np.full_like(data_3d, np.nan)
        
    def _calculate_global_prior(self):
        """
        Calculates a global pooled mean for robust starting points (t=0 imputation).
        """
        initial_data = self.data_3d[:, 0, :]
        global_mean = np.nanmean(initial_data, axis=0) 
        
        for i in range(self.n_entities):
            for k in range(self.n_fields):
                # Impute the t=0 start point with the global mean if missing
                if np.isnan(self.data_3d[i, 0, k]):
                    self.smoothed_data[i, 0, k] = global_mean[k]
                else:
                    self.smoothed_data[i, 0, k] = self.data_3d[i, 0, k]

    def run_temporal_smoothing(self, smoothing_level=0.3, seasonal_periods=5):
        """
        Runs Holt-Winters (Triple Exponential Smoothing) for temporal modeling.
        """
        self._calculate_global_prior()
        
        # NOTE: Using a single tqdm bar for the estimation process
        for i in tqdm(range(self.n_entities), desc="Processing entities (Holt-Winters)"):
            for k in range(self.n_fields):
                obs = self.data_3d[i, :, k]
                series = pd.Series(obs)
                
                # 1. Temporal Smoothing (Holt-Winters)
                try:
                    # Model requires no NaNs, so we impute with the initial estimate first for the fit
                    initial_imputation = series.fillna(self.smoothed_data[i, 0, k])
                    
                    model_fit = ExponentialSmoothing(
                        initial_imputation, 
                        seasonal_periods=seasonal_periods, 
                        trend='add', 
                        seasonal=None, # Set to None for faster, more stable Holt-Winters
                        initialization_method="estimated"
                    ).fit(smoothing_level=smoothing_level, use_boxcox=False)
                    
                    smoothed_values = model_fit.fittedvalues.values
                    
                    # 2. Confidence Interval (95% Estimate)
                    # Use model residuals to estimate variance (more rigorous than simple rolling std)
                    residuals = initial_imputation.values - smoothed_values
                    ci_scale = np.nanstd(residuals) * 1.96 
                    
                    # Store the smoothed mean
                    self.smoothed_data[i, :, k] = smoothed_values
                    
                    # Store the confidence interval
                    self.hdi_3[i, :, k] = smoothed_values - ci_scale
                    self.hdi_97[i, :, k] = smoothed_values + ci_scale

                except Exception as e:
                    # Fallback to simple mean if the complex model fails (e.g., all NaNs or bad data)
                    mean_val = np.nanmean(obs)
                    std_val = np.nanstd(obs)
                    self.smoothed_data[i, :, k] = np.full(self.n_timesteps, mean_val)
                    self.hdi_3[i, :, k] = mean_val - 1.96 * std_val
                    self.hdi_97[i, :, k] = mean_val + 1.96 * std_val
        
        print("Holt-Winters smoothing and estimation complete.")


# --- Worker Function for Parallel Processing ---

def _run_holtwinter_chunk(inputs: Tuple[str, List[str]]) -> Optional[pd.DataFrame]:
    """
    Worker function to process a single chunk of entities using the Holt-Winters estimator.
    """
    data_file, entity_subset = inputs
    
    try:
        # Load and prepare data for the specific chunk
        data_3d, entities, times, fields, original_df, _ = load_and_prepare_data(data_file, entity_subset)
        
        # Instantiate and run the classical estimator
        estimator = HoltWintersEstimator(data_3d, entities, times, fields)
        estimator.run_temporal_smoothing()
        
        # Extract and format results for this chunk
        results_df = extract_and_format_results(estimator, original_df, entities, times, fields)
        return results_df
        
    except Exception as e:
        print(f"\nFATAL ERROR in chunk processing (Entities {entities[0]} to {entities[-1]}): {e}", file=sys.stderr)
        return None

# --- Utility Functions (Load, Extract, Generate) ---

def load_and_prepare_data(data_file: str, entities_subset: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], List[Any], List[str], pd.DataFrame, np.ndarray]:
    """
    Loads the input data CSV, handles subsampling/subsetting, and structures it into a 3D NumPy array.
    """
    
    print(f"Loading data from '{data_file}'...")
    try:
        df = pd.read_csv(data_file, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    if len(df.columns) < 3:
        print("Error: Input CSV must have at least 3 columns (Entity, Time, Field).", file=sys.stderr)
        sys.exit(1)

    # 1. Automatic Column Assignment
    entity_col = df.columns[0]
    time_col = df.columns[1]
    
    # --- Subsampling/Subsetting Step ---
    unique_entities = df[entity_col].unique()
    
    if entities_subset is not None:
        # Use only the chunk defined by parallel-run
        df = df[df[entity_col].isin(entities_subset)].copy()
        print(f"Subset: Processing {len(entities_subset)} entities for this chunk.")
    
    # --- Automatic Filtering of Non-Numerical Field Columns ---
    potential_field_cols = df.columns[2:]
    numeric_check_df = df[potential_field_cols].apply(pd.to_numeric, errors='coerce')
    field_cols = numeric_check_df.columns[numeric_check_df.notna().any()].to_list()
    
    dropped_cols = [col for col in potential_field_cols if col not in field_cols]
    if dropped_cols:
        print(f"\n--- WARNING: Dropped Non-Numeric Field Columns ---")
        print(f"The following columns were dropped from the model because they contain non-numerical data (e.g., dates/text): {dropped_cols}")
        print(f"Only numerical fields can be used for modeling.")
        print("--------------------------------------------------\n")

    if not field_cols:
        print("Error: No valid numerical field columns found (Column 3 onwards).", file=sys.stderr)
        sys.exit(1)
        
    print(f"Assigned Columns: Entity='{entity_col}', Time='{time_col}'. Valid Numeric Fields: {field_cols}")
    
    # 2. Convert Time to Sequential Integers if necessary
    if not pd.api.types.is_integer_dtype(df[time_col]):
        print(f"Warning: Time column '{time_col}' is not integer. Converting to sequential steps.")
        df[time_col] = df.groupby(entity_col)[time_col].transform(lambda x: pd.factorize(x)[0] + 1)
        
    df_processed = df.copy() 
    
    # 3. Pivot to Wide Format (Entity x Time) for all fields
    pivoted_dict = {}
    
    print("\nPreparing 3D data structure...")
    for field_col in tqdm(field_cols, desc="Pivoting fields"):
        try:
            pivoted_df = df.pivot(index=entity_col, columns=time_col, values=field_col)
            pivoted_dict[field_col] = pivoted_df
        except Exception as e:
            print(f"\nError pivoting data for {field_col}: {e}. Check for non-sequential time steps.", file=sys.stderr)
            sys.exit(1)

    # 4. Consolidate into 3D Array
    entity_names = pivoted_dict[field_cols[0]].index.to_list()
    time_index = pivoted_dict[field_cols[0]].columns.to_list()
    
    n_entities = len(entity_names)
    n_timesteps = len(time_index)
    n_fields = len(field_cols)
    
    # Final data structure: (Entity, Time, Field)
    observed_data_3d = np.zeros((n_entities, n_timesteps, n_fields))
    observed_data_3d[:] = np.nan # Initialize with NaN
    
    for k, field_col in enumerate(field_cols):
        observed_data_3d[:, :, k] = pivoted_dict[field_col].values

    # NOTE: The Holt-Winters estimator does not require initial_values in the same way MCMC did.
    # We return a dummy array for the initial_values parameter.
    initial_values = np.zeros((n_entities, n_timesteps, n_fields))
        
    print(f"\nData prepared: {n_entities} entities, {n_timesteps} time steps, {n_fields} fields.")
    
    return observed_data_3d, entity_names, time_index, field_cols, df_processed, initial_values

def extract_and_format_results(estimator: HoltWintersEstimator, original_df: pd.DataFrame, entities: List[str], times: List[Any], fields: List[str]) -> pd.DataFrame:
    """
    Extracts the estimated mean and confidence intervals from the Holt-Winters output 
    and merges them back into the original input DataFrame.
    """
    print("Extracting and consolidating results...")
        
    # Extraction is always from the HoltWintersEstimator object
    true_value_mean = estimator.smoothed_data
    true_value_hdi_3 = estimator.hdi_3
    true_value_hdi_97 = estimator.hdi_97
    
    results = []
    
    n_entities, n_timesteps, n_fields = true_value_mean.shape
    
    entity_col_name = original_df.columns[0]
    time_col_name = original_df.columns[1]

    # --- PROGRESS BAR FOR RESULT EXTRACTION ---
    print("Formatting and merging estimates...")
    # NOTE: We only merge back the sampled entities. The rest of the original data remains untouched.
    
    for i in tqdm(range(n_entities), desc="Extracting entity data"):
        for j in range(n_timesteps):
            row = {
                entity_col_name: entities[i], 
                time_col_name: times[j],
            }
            
            for k in range(n_fields):
                field_col = fields[k]
                row.update({
                    f"{field_col}_true_value_mean": true_value_mean[i, j, k], 
                    f"{field_col}_hdi_3": true_value_hdi_3[i, j, k], 
                    f"{field_col}_hdi_97": true_value_hdi_97[i, j, k]
                })
            results.append(row)
            
    estimates_df = pd.DataFrame(results)
    
    # Merge estimates with the original processed DataFrame
    # Note: This merge will only affect the rows corresponding to the sampled entities
    final_output_df = pd.merge(
        original_df,
        estimates_df,
        on=[entity_col_name, time_col_name],
        how='left' 
    )
    
    print("Consolidation complete. Estimates merged back into original data structure.")
    return final_output_df

def generate_dummy_data(data_file: str = "dummy_data.csv"):
    """
    Generates a dummy CSV file to test the multi-variate script.
    """
    print(f"Generating dummy data at '{data_file}' with multiple, correlated fields...")
    N_TIMESTEPS = 150
    N_ENTITIES = 5
    times = np.arange(N_TIMESTEPS)
    
    data = []
    
    for i in range(N_ENTITIES):
        entity_name = f"reactor_{i}"
        
        # Base process for all fields
        base_signal = (i+1) * 5 * np.sin((0.1 + i*0.02) * times)
        
        # Field 1: Plasma Temp (Correlates strongly with base signal)
        obs_temp = base_signal * 1.5 + 50 + np.random.normal(scale=2.0, size=N_TIMESTEPS)
        
        # Field 2: Density (Correlates negatively with base signal, lags slightly)
        obs_density = -base_signal * 1.2 + 20 + np.random.normal(scale=1.5, size=N_TIMESTEPS)

        # Field 3: Status_Code (More noise, lower correlation)
        obs_status = np.abs(base_signal) + np.random.normal(scale=3.0, size=N_TIMESTEPS)
        
        # Introduce massive sparsity/missingness
        missing_indices = np.random.choice(N_TIMESTEPS, size=int(N_TIMESTEPS * 0.6), replace=False)
        
        # Format as long-data records
        for t in range(N_TIMESTEPS):
            row = {
                "reactor_id": entity_name,
                "day": t,
                "plasma_temp": obs_temp[t],
                "density": obs_density[t],
                "status_code": obs_status[t]
            }
            # Apply sparsity (missing values)
            if t in missing_indices:
                # If time t is missing, we randomly make one or two fields missing.
                fields_to_nan = np.random.choice(["plasma_temp", "density", "status_code"], 
                                                 size=np.random.randint(1, 4), replace=False)
                for field in fields_to_nan:
                    row[field] = np.nan
            
            data.append(row)
            
    df = pd.DataFrame(data)
    df.to_csv(data_file, index=False)
    print("Dummy data generation complete. Fields: plasma_temp, density, status_code")


# --- Main execution with command-line arguments ---
def main():
    parser = argparse.ArgumentParser(
        description="V2 Stable Classical Estimator: Cleans and fills data using the fast Holt-Winters method.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- "run" command (The fast single-core estimation method) ---
    run_parser = subparsers.add_parser(
        "run", 
        help="Run the stable Holt-Winters inference on a data file.",
        description="Saves stable Holt-Winters estimates to [input_file]_gaps_filled.csv (Single Core)."
    )
    run_parser.add_argument("data_file", type=str,
                            help="Path to the input CSV file (long format).")

    # --- "parallel-run" command (The massive, whole-dataset method) ---
    parallel_parser = subparsers.add_parser(
        "parallel-run", 
        help="Runs Holt-Winters on the full dataset using multiprocessing.",
        description="Divides the entire dataset across multiple cores for fast processing of all entities."
    )
    parallel_parser.add_argument("data_file", type=str,
                                help="Path to the input CSV file (long format).")
    parallel_parser.add_argument("--num-cores", type=int, default=8,
                                help="Number of CPU cores to use for parallel processing (default: 8).")
    
    # --- "make-dummy-data" command ---
    dummy_parser = subparsers.add_parser(
        "make-dummy-data", 
        help="Generate a dummy CSV file to test the script.",
        description="Creates 'dummy_data.csv' with 5 entities, 150 time steps, and multiple correlated fields."
    )
    dummy_parser.add_argument("--file-name", type=str, default="dummy_data.csv",
                              help="Name of the dummy file to create.")
    
    args = parser.parse_args()

    # --- Execute the chosen command ---

    if args.command == "run":
        print(f"--- Running V2 Classical Estimator (Single Core) ---")
        base, ext = os.path.splitext(args.data_file)
        output_file = f"{base}_gaps_filled{ext}"

        # 1. Load data and structure into 3D array (Entire dataset)
        # We pass None for entities_subset to process all
        data_3d, entities, times, fields, original_df, _ = load_and_prepare_data(args.data_file, entities_subset=None)
        
        # 2. Instantiate and run the classical estimator
        estimator = HoltWintersEstimator(data_3d, entities, times, fields)
        estimator.run_temporal_smoothing()

        # 3. Extract and save results
        results_df = extract_and_format_results(estimator, original_df, entities, times, fields)

    elif args.command == "parallel-run":
        print(f"--- Running V2 Parallel Classical Estimator across {args.num_cores} cores ---")
        
        # 1. Get ALL unique entities
        full_df = pd.read_csv(args.data_file, low_memory=False)
        all_entities = full_df[full_df.columns[0]].unique()
        n_entities = len(all_entities)
        
        # Determine chunk size
        num_cores = args.num_cores
        chunk_size = int(np.ceil(n_entities / num_cores))
        entity_chunks = [all_entities[i:i + chunk_size].tolist() for i in range(0, n_entities, chunk_size)]
        
        # Prepare inputs for multiprocessing pool
        pool_inputs = [(args.data_file, chunk) for chunk in entity_chunks]
        
        # Execute the pool
        print(f"Dividing {n_entities} entities into {len(entity_chunks)} chunks of size approx {chunk_size}...")
        pool = multiprocessing.Pool(processes=num_cores)
        
        # Run the classical estimator on each chunk
        # NOTE: We use imap for iterable progress bar feedback
        chunk_results = list(tqdm(pool.imap(_run_holtwinter_chunk, pool_inputs), total=len(pool_inputs), desc="Total Parallel Processing"))
        
        pool.close()
        pool.join()
        
        # Merge all results
        print("\nMerging results from all cores...")
        # Concatenate all non-None DataFrames from the chunks
        results_df = pd.concat([res for res in chunk_results if res is not None], ignore_index=True)
        
        base, ext = os.path.splitext(args.data_file)
        output_file = f"{base}_gaps_filled{ext}"

    elif args.command == "make-dummy-data":
        generate_dummy_data(args.file_name)
        return

    try:
        if args.command == "run" or args.command == "parallel-run":
            results_df.to_csv(output_file, index=False)
            print(f"\n--- Run Complete. Cleaned data saved to: {output_file} ---")
    except Exception as e:
        print(f"Error saving final consolidated file: {e}", file=sys.stderr)
        
if __name__ == "__main__":
    main()
