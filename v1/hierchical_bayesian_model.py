import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import argparse
import sys
import os

print(f"Running on PyMC v{pm.__version__}")

def load_and_prepare_data(data_file, entity_col, time_col, field_col):
    """
    Loads the input data CSV, pivots it to a (N_Entities x N_Timesteps)
    matrix, and prepares it for PyMC.
    """
    print(f"Loading data from '{data_file}'...")
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Check for required columns
    required_cols = [entity_col, time_col, field_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in data: {missing_cols}", file=sys.stderr)
        sys.exit(1)
        
    print("Pivoting data to (Entities x Time) matrix...")
    
    # Pivot to wide format: (rows=entities, cols=time, values=field)
    try:
        pivoted_df = df.pivot(index=entity_col, columns=time_col, values=field_col)
    except Exception as e:
        print(f"Error pivoting data. Do you have duplicate (entity, time) entries? Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Get entity names and time index for later use
    entity_names = pivoted_df.index.to_list()
    time_index = pivoted_df.columns.to_list()
    
    # Convert to NumPy array. PyMC works best with this.
    # NaNs are expected and handled.
    observed_data_matrix = pivoted_df.values
    
    n_entities, n_timesteps = observed_data_matrix.shape
    print(f"Data prepared: {n_entities} entities, {n_timesteps} time steps.")
    
    return observed_data_matrix, entity_names, time_index

def define_pooled_hbm(data_matrix):
    """
    Defines the Hierarchical Bayesian Model (State-Space) in PyMC.
    """
    n_entities, n_timesteps = data_matrix.shape
    
    print("Defining Pooled Hierarchical Bayesian Model...")
    
    coords = {"entity": np.arange(n_entities), "time": np.arange(n_timesteps)}
    
    with pm.Model(coords=coords) as hbm_model:
        # --- Hierarchical Priors (The "Pooling" part) ---
        # We assume all entities are "similar". We learn the global
        # average noise levels.
        
        # Global prior for process noise (how much the state "jumps" per step)
        mu_sigma_proc = pm.HalfNormal("mu_sigma_proc", sigma=1.0)
        sigma_sigma_proc = pm.HalfNormal("sigma_sigma_proc", sigma=0.5)
        
        # Global prior for observation noise (how noisy the sensors are)
        mu_sigma_obs = pm.HalfNormal("mu_sigma_obs", sigma=2.0)
        sigma_sigma_obs = pm.HalfNormal("sigma_sigma_obs", sigma=0.5)

        # --- Entity-Level Parameters ---
        # Each entity gets its *own* noise level, drawn from the global prior.
        
        # Process noise for each entity
        sigma_proc = pm.HalfNormal("sigma_proc", sigma=sigma_sigma_proc, mu=mu_sigma_proc, dims="entity")
        
        # Observation noise for each entity
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=sigma_sigma_obs, mu=mu_sigma_obs, dims="entity")

        # --- State-Space Model (Temporal Transitions) ---
        
        # 'true_value' is the hidden, unobserved state we're estimating.
        # We model it as a GaussianRandomWalk (e.g., state[t] ~ N(state[t-1], sigma_proc))
        # This is a (N_Entities x N_Timesteps) matrix.
        # The 'dims' parameter is key for broadcasting sigma_proc correctly.
        true_value = pm.GaussianRandomWalk("true_value", 
                                           sigma=sigma_proc[:, None], 
                                           dims=("entity", "time"))

        # --- Likelihood (Connecting Model to Data) ---
        # This is where we tell the model:
        # "The 'observed_data' is just the 'true_value' plus 'observation_noise'."
        # PyMC automatically handles the NaNs (missing values) in data_matrix.
        likelihood = pm.Normal("likelihood", 
                               mu=true_value, 
                               sigma=sigma_obs[:, None], 
                               observed=data_matrix, 
                               dims=("entity", "time"))
        
    print("Model definition complete.")
    return hbm_model

def run_inference(hbm_model, draws=1000, tune=1000):
    """
    Runs the MCMC simulation to fit the model to the data.
    """
    print(f"Starting MCMC inference (draws={draws}, tune={tune})... This will take time.")
    with hbm_model:
        # Use NUTS sampler
        idata = pm.sample(draws=draws, tune=tune, chains=4, cores=1)
    
    print("Inference complete.")
    return idata

def save_results(idata, entity_names, time_index, output_file):
    """
    Extracts the mean and HDI (confidence interval) for the 'true_value'
    and saves them to a long-format CSV.
    """
    print(f"Extracting results and saving to '{output_file}'...")
    
    # Extract the posterior mean and 94% HDI (Highest Density Interval)
    # This gives us the "best guess" and the "confidence interval"
    true_value_mean = idata.posterior["true_value"].mean(dim=("chain", "draw")).values
    true_value_hdi = az.hdi(idata.posterior["true_value"], hdi_prob=0.94).x.values
    
    results = []
    
    n_entities, n_timesteps = true_value_mean.shape
    
    for i in range(n_entities):
        for j in range(n_timesteps):
            results.append({
                "entity": entity_names[i],
                "time": time_index[j],
                "true_value_mean": true_value_mean[i, j],
                "hdi_3": true_value_hdi[i, j, 0], # 3% percentile
                "hdi_97": true_value_hdi[i, j, 1]  # 97% percentile
            })
            
    results_df = pd.DataFrame(results)
    
    try:
        results_df.to_csv(output_file, index=False)
        print(f"Successfully saved V2 estimates to '{output_file}'.")
    except Exception as e:
        print(f"Error saving results: {e}", file=sys.stderr)
        
    return results_df

def plot_results(observed_matrix, results_df, entity_names, time_index, plot_prefix):
    """
    Generates and saves plots for each entity's estimated true value.
    """
    print(f"Generating diagnostic plots with prefix '{plot_prefix}'...")
    
    n_entities = len(entity_names)
    
    for i in range(n_entities):
        entity = entity_names[i]
        
        # Get data for this entity
        entity_obs = observed_matrix[i, :]
        entity_results = results_df[results_df['entity'] == entity].sort_values(by="time")
        
        plt.figure(figsize=(15, 6))
        
        # Plot 94% HDI (Confidence Interval)
        plt.fill_between(
            entity_results['time'],
            entity_results['hdi_3'],
            entity_results['hdi_97'],
            color='C0', alpha=0.3, label="94% HDI (Confidence)"
        )
        
        # Plot Posterior Mean (Best Guess "True Value")
        plt.plot(
            entity_results['time'],
            entity_results['true_value_mean'],
            color='C0', lw=2,
            label="HBM 'True Value' (Mean)"
        )
        
        # Plot the original, noisy, sparse observations
        plt.plot(
            time_index,
            entity_obs,
            'x', color='C3', ms=6, alpha=0.7,
            label="Noisy Observations (V1 Data)"
        )
        
        plt.title(f"V2 Estimate: '{entity}'")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plot_filename = f"{plot_prefix}_{entity}.png"
        try:
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}", file=sys.stderr)
        plt.close()

def generate_dummy_data(data_file="dummy_data.csv"):
    """
    Generates a dummy CSV file to test the script.
    """
    print(f"Generating dummy data at '{data_file}'...")
    N_TIMESTEPS = 150
    N_ENTITIES = 5
    times = np.arange(N_TIMESTEPS)
    
    data = []
    
    for i in range(N_ENTITIES):
        entity_name = f"reactor_{i}"
        
        # Create a unique "ground truth" for each entity
        ground_truth = (i+1) * 5 * np.sin((0.1 + i*0.02) * times)
        
        # Create "true state" with some process noise
        true_state = np.zeros(N_TIMESTEPS)
        for t in range(1, N_TIMESTEPS):
            true_state[t] = true_state[t-1] * 0.9 + ground_truth[t] + np.random.normal(scale=1.0)
            
        # Create "observations" with sensor noise
        sensor_noise = np.random.normal(scale=1.5 + i*0.5, size=N_TIMESTEPS)
        observations = true_state + sensor_noise
        
        # Make 40% of the data missing (NaN)
        missing_indices = np.random.choice(N_TIMESTEPS, size=int(N_TIMESTEPS * 0.4), replace=False)
        observations[missing_indices] = np.nan
        
        # Format as long-data records
        for t in range(N_TIMESTEPS):
            data.append({
                "reactor_id": entity_name,
                "day": t,
                "plasma_temp": observations[t]
            })
            
    df = pd.DataFrame(data)
    df.to_csv(data_file, index=False)
    print("Dummy data generation complete.")


# --- Main execution with command-line arguments ---
def main():
    parser = argparse.ArgumentParser(
        description="V2 Bayesian Estimator: Cleans and fills V1 data using a pooled HBM.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- "run" command ---
    run_parser = subparsers.add_parser(
        "run", 
        help="Run the Bayesian inference on a data file.",
        description="Reads a CSV, pivots it, defines and runs the HBM, and saves the clean results."
    )
    run_parser.add_argument("--data-file", type=str, required=True,
                            help="Path to the input CSV file (long format).")
    run_parser.add_argument("--entity-col", type=str, required=True,
                            help="Column name for the entity ID (e.g., 'reactor_id')")
    run_parser.add_argument("--time-col", type=str, required=True,
                            help="Column name for the time step (e.g., 'day')")
    run_parser.add_argument("--field-col", type=str, required=True,
                            help="The noisy, sparse field to analyze (e.g., 'plasma_temp')")
    run_parser.add_argument("--output-file", type=str, default="hbm_estimates.csv",
                            help="Path to save the resulting estimates CSV.")
    run_parser.add_argument("--plot-prefix", type=str, default="hbm_plot",
                            help="Prefix for saving diagnostic plots (e.g., 'hbm_plot_reactor_0.png')")
    run_parser.add_argument("--draws", type=int, default=1000,
                            help="Number of MCMC draws.")
    run_parser.add_argument("--tune", type=int, default=1000,
                            help="Number of MCMC tuning steps.")

    # --- "make-dummy-data" command ---
    dummy_parser = subparsers.add_parser(
        "make-dummy-data", 
        help="Generate a dummy CSV file to test the script.",
        description="Creates 'dummy_data.csv' with 5 entities and 150 time steps."
    )
    dummy_parser.add_argument("--file-name", type=str, default="dummy_data.csv",
                              help="Name of the dummy file to create.")
    
    args = parser.parse_args()

    # --- Execute the chosen command ---

    if args.command == "run":
        print("--- Running V2 Bayesian Estimator ---")
        # 1. Load data
        observed_matrix, entities, times = load_and_prepare_data(
            args.data_file, args.entity_col, args.time_col, args.field_col
        )
        # 2. Define model
        hbm = define_pooled_hbm(observed_matrix)
        # 3. Run model
        idata = run_inference(hbm, draws=args.draws, tune=args.tune)
        # 4. Save results
        results_df = save_results(idata, entities, times, args.output_file)
        # 5. Plot results
        plot_results(observed_matrix, results_df, entities, times, args.plot_prefix)
        
        print("\n--- V2 Run Complete ---")
        print(f"Estimates saved to: {args.output_file}")
        print(f"Plots saved with prefix: {args.plot_prefix}")

    elif args.command == "make-dummy-data":
        generate_dummy_data(args.file_name)

if __name__ == "__main__":
    # Example of how to run this script:
    #
    # 1. Generate test data:
    #    python hierarchical_bayesian_model.py make-dummy-data
    #
    # 2. Run the analysis on the dummy data:
    #    python hierarchical_bayesian_model.py run \
    #      --data-file dummy_data.csv \
    #      --entity-col reactor_id \
    #      --time-col day \
    #      --field-col plasma_temp \
    #      --output-file hbm_estimates.csv \
    #      --plot-prefix my_hbm_run
    
    main()

