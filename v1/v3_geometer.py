import pandas as pd
import numpy as np
import umap
import hdbscan
import plotly.express as px
import argparse
import sys
from sklearn.preprocessing import StandardScaler

print("V3/V4 Geometer Script Initialized (Temporal).")

# --- Functions for 'run' command (Temporal Embedding) ---

def load_and_prepare_data_temporal(input_file, entity_col, time_col, field_col, window_size, step_size):
    """
    Loads the *clean* V2 estimates and uses a SLIDING WINDOW
    to create a (N_Samples x N_Timesteps) matrix for temporal embedding.
    
    Each "sample" is a vector representing the behavior of one entity
    over a short period (`window_size`).
    """
    print(f"Loading V2 estimates from '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: V2 estimates file '{input_file}' not found.", file=sys.stderr)
        print("Please run the V2 'hierarchical_bayesian_model.py' script first.", file=sys.stderr)
        sys.exit(1)

    # Check for required columns from the V2 output
    required_cols = [entity_col, time_col, field_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in V2 data: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    print(f"Vectorizing with sliding window (window_size={window_size}, step_size={step_size})...")
    
    vector_list = []
    metadata_list = [] # To track (entity, end_time)

    # Group by entity to process each one's time-series individually
    for entity, group_df in df.groupby(entity_col):
        
        # Ensure data is sorted by time for correct windowing
        group_df = group_df.sort_values(by=time_col)
        time_series = group_df[field_col].values
        times = group_df[time_col].values
        
        n_timesteps = len(time_series)
        
        if n_timesteps < window_size:
            print(f"Warning: Entity '{entity}' has only {n_timesteps} steps, less than window size {window_size}. Skipping.", file=sys.stderr)
            continue
            
        # Slide the window across the time-series
        for i in range(0, n_timesteps - window_size + 1, step_size):
            
            window_end_index = i + window_size
            window_vector = time_series[i:window_end_index]
            
            # Record the high-dimensional vector
            vector_list.append(window_vector)
            
            # Record its "metadata" (which entity, at what time)
            metadata_list.append({
                "entity": entity,
                "time": times[window_end_index - 1] # Label vector by its end time
            })

    if not vector_list:
        print("Error: No vectors created. Is --window-size larger than your time-series?", file=sys.stderr)
        sys.exit(1)
        
    high_dim_vectors = np.array(vector_list)
    
    # Create a DataFrame from the metadata
    metadata_df = pd.DataFrame(metadata_list)
    
    print(f"Vectorized data: {high_dim_vectors.shape[0]} samples (slices), {high_dim_vectors.shape[1]} dimensions (window size).")
    
    return high_dim_vectors, metadata_df

def preprocess_data(data):
    """
    Scales the data so that each dimension (time step in the window)
    has zero mean and unit variance. This is critical for UMAP.
    """
    print("Scaling data (StandardScaler)...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

def run_umap(data, n_neighbors=15, min_dist=0.1, n_components=3, random_state=42):
    """
    This is the "Geometer." It runs UMAP to perform
    non-linear dimensionality reduction (the "un-crumpling").
    """
    print(f"Running UMAP (n_components={n_components}, n_neighbors={n_neighbors})... This may take a moment.")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,  # Controls local (low) vs. global (high) structure
        min_dist=min_dist,        # Controls how tightly UMAP packs points together
        n_components=n_components,# 2 for 2D, 3 for 3D
        random_state=random_state,
    )
    
    embedding = reducer.fit_transform(data)
    print("UMAP embedding complete.")
    return embedding

def save_embedding(metadata_df, embedding, output_file):
    """
    Saves the raw UMAP embedding results (entity, time, x, y, z) to a CSV.
    This is the main artifact of the 'run' command.
    """
    print(f"Saving raw temporal embedding to '{output_file}'...")
    
    if embedding.shape[1] == 3:
        dims = ['x', 'y', 'z']
    elif embedding.shape[1] == 2:
        dims = ['x', 'y']
    else:
        dims = [f'dim_{i}' for i in range(embedding.shape[1])]
        
    embedding_df = pd.DataFrame(embedding, columns=dims)
    
    # Combine metadata (entity, time) with embedding (x, y, z)
    results_df = pd.concat([metadata_df, embedding_df], axis=1)
    
    try:
        results_df.to_csv(output_file, index=False)
        print(f"Save complete. This file is the input for the 'explore' command.")
    except Exception as e:
        print(f"Error saving output file: {e}", file=sys.stderr)

# --- Functions for 'explore' command ---

def load_embedding(input_file):
    """
    Loads the pre-calculated temporal embedding file from 'run'.
    """
    print(f"Loading temporal embedding from '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Embedding file '{input_file}' not found.", file=sys.stderr)
        print("Did you run the 'run' command first?")
        sys.exit(1)
    
    # Infer dimensions from columns
    dims = [col for col in df.columns if col not in ['entity', 'time']]
    print(f"Found {len(dims)} dimensions: {dims}")
    
    embedding = df[dims].values
    return df, embedding, dims

def find_structure(embedding, min_cluster_size=10):
    """
    This is the "Query" engine. It runs HDBSCAN to find
    density-based clusters (geometric regularities / "states") in the map.
    """
    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size}) to find structure...")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        gen_min_span_tree=True
    )
    
    labels = clusterer.fit_predict(embedding)
    print(f"Found {len(np.unique(labels))} clusters (states) (including noise points labeled -1).")
    return labels

def save_clustered_results(results_df, output_file):
    """
    Saves the final map *with cluster labels* to a new CSV.
    """
    print(f"Saving clustered results to '{output_file}'...")
    try:
        results_df.to_csv(output_file, index=False)
        print("Save complete.")
    except Exception as e:
        print(f"Error saving output file: {e}", file=sys.stderr)

def plot_interactive_states(results_df, dims, plot_file):
    """
    Generates an interactive plot of the "States" (colored by cluster).
    This shows the "regions" or "microlaws" of your manifold.
    """
    print(f"Generating interactive 'states' plot at '{plot_file}'...")
    
    results_df['cluster_label'] = results_df['cluster_label'].astype(str)
    
    title = "Interactive Manifold of Temporal States (Microlaws)"
    
    if len(dims) == 3:
        fig = px.scatter_3d(
            results_df, x='x', y='y', z='z',
            color='cluster_label',
            hover_data=['entity', 'time'],
            title=title, template="plotly_dark"
        )
    else: # 2D
        fig = px.scatter(
            results_df, x='x', y='y',
            color='cluster_label',
            hover_data=['entity', 'time'],
            title=title, template="plotly_dark"
        )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(width=1200, height=800, legend_title="Discovered States (Clusters)")
    
    try:
        fig.write_html(plot_file)
        print("Interactive 'states' plot saved.")
    except Exception as e:
        print(f"Error saving interactive plot: {e}", file=sys.stderr)

def plot_interactive_trajectories(results_df, dims, plot_file):
    """
    Generates an interactive plot of the "Trajectories" (colored by entity).
    This shows the *path* each entity took over time.
    """
    print(f"Generating interactive 'trajectories' plot at '{plot_file}'...")
    
    # Sort by time to make the lines connect correctly
    results_df = results_df.sort_values(by=['entity', 'time'])
    
    title = "Interactive Manifold of Entity Trajectories"
    
    if len(dims) == 3:
        fig = px.line_3d(
            results_df, x='x', y='y', z='z',
            color='entity',
            markers=True,
            hover_data=['time', 'cluster_label'],
            title=title, template="plotly_dark"
        )
    else: # 2D
        fig = px.line(
            results_df, x='x', y='y',
            color='entity',
            markers=True,
            hover_data=['time', 'cluster_label'],
            title=title, template="plotly_dark"
        )
        
    fig.update_traces(marker=dict(size=3), line=dict(width=1))
    fig.update_layout(width=1200, height=800, legend_title="Entity")
    
    try:
        fig.write_html(plot_file)
        print("Interactive 'trajectories' plot saved.")
    except Exception as e:
        print(f"Error saving interactive plot: {e}", file=sys.stderr)

# --- Main execution ---

def main():
    parser = argparse.ArgumentParser(description="V3/V4 Geometer: Run & explore TEMPORAL manifold embeddings.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # --- "run" command parser ---
    run_parser = subparsers.add_parser(
        "run", 
        help="1. Generate the temporal UMAP embedding from V2 data (slow step).",
        description="Reads V2 HBM estimates, uses a sliding window to vectorize, and runs UMAP."
    )
    run_parser.add_argument("--input-file", required=True, 
                            help="Path to the V2 estimates CSV (output from hbm script).")
    run_parser.add_argument("--entity-col", required=True, 
                            help="Column name for the entity ID (e.g., 'entity')")
    run_parser.add_argument("--time-col", required=True, 
                            help="Column name for the time step (e.g., 'time')")
    run_parser.add_argument("--field-col", required=True, 
                            help="The *clean* V2 field to analyze (e.g., 'true_value_mean')")
    run_parser.add_argument("--output-embedding-file", default="v4_temporal_embedding.csv", 
                            help="Path to save the new temporal UMAP embedding CSV.")
    # --- Args for 3D and Time ---
    run_parser.add_argument("--n-components", type=int, default=3,
                            help="[UMAP] Number of dimensions for the embedding (e.g., 2 or 3).")
    run_parser.add_argument("--window-size", type=int, default=30,
                            help="[Window] Number of time steps to include in each vector.")
    run_parser.add_argument("--step-size", type=int, default=5,
                            help="[Window] Number of time steps to slide the window forward.")
    # --- UMAP Tuning ---
    run_parser.add_argument("--n-neighbors", type=int, default=15,
                            help="[UMAP] Controls local vs. global structure. Try values from 5-50.")
    run_parser.add_argument("--min-dist", type=float, default=0.1,
                            help="[UMAP] Controls how tightly points are packed. Try values from 0.0-1.0.")

    # --- "explore" command parser ---
    explore_parser = subparsers.add_parser(
        "explore", 
        help="2. Interact with an existing temporal embedding (cluster and plot, fast step).",
        description="Reads the temporal embedding, runs HDBSCAN, and generates interactive HTML plots."
    )
    explore_parser.add_argument("--input-embedding-file", required=True,
                                help="Path to the temporal embedding CSV from 'run' (e.g., 'v4_temporal_embedding.csv').")
    explore_parser.add_argument("--output-clustered-file", default="v4_clustered_results.csv",
                                help="Path to save the final results (embedding + cluster labels).")
    explore_parser.add_argument("--plot-prefix", default="v4_manifold_plot", 
                                help="Prefix for plot files. Will generate '<prefix>_states.html' and '<prefix>_trajectories.html'.")
    # HDBSCAN Tuning
    explore_parser.add_argument("--min-cluster-size", type=int, default=10,
                                help="[HDBSCAN] Min entities/slices to form a cluster. Increase for less, smaller clusters.")

    args = parser.parse_args()

    # --- Execute the chosen command ---

    if args.command == "run":
        print("--- Running V4 'run' command (Temporal Embedding) ---")
        high_dim_vectors, metadata_df = load_and_prepare_data_temporal(
            args.input_file, args.entity_col, args.time_col, args.field_col,
            args.window_size, args.step_size
        )
        data_scaled = preprocess_data(high_dim_vectors)
        embedding = run_umap(
            data_scaled, n_neighbors=args.n_neighbors, min_dist=args.min_dist, n_components=args.n_components
        )
        save_embedding(metadata_df, embedding, args.output_embedding_file)
        
        print(f"\n--- 'run' complete. Temporal embedding saved to: {args.output_embedding_file} ---")
        print(f"Next step: Run the 'explore' command on this file.")

    elif args.command == "explore":
        print("--- Running V4 'explore' command ---")
        embedding_df, embedding, dims = load_embedding(args.input_embedding_file)
        labels = find_structure(embedding, min_cluster_size=args.min_cluster_size)
        embedding_df['cluster_label'] = labels
        save_clustered_results(embedding_df, args.output_clustered_file)
        
        # Plot both views
        plot_interactive_states(embedding_df, dims, f"{args.plot_prefix}_states.html")
        plot_interactive_trajectories(embedding_df, dims, f"{args.plot_prefix}_trajectories.html")
        
        print(f"\n--- 'explore' complete. Clustered data saved to: {args.output_clustered_file} ---")
        print(f"Interactive plots saved as '{args.plot_prefix}_states.html' and '{args.plot_prefix}_trajectories.html'")

if __name__ == "__main__":
    # Example of how to run this script from the command line:
    #
    # 0. (Run V2 script first to get 'hbm_estimates.csv')
    #    python hierarchical_bayesian_model.py run --data-file dummy_data.csv ... --output-file hbm_estimates.csv
    #
    # 1. Then, run the V4 'run' command to generate the 3D embedding:
    #    python v3_geometer.py run \
    #      --input-file hbm_estimates.csv \
    #      --entity-col entity \
    #      --time-col time \
    #      --field-col true_value_mean \
    #      --output-embedding-file my_temporal_embedding.csv \
    #      --n-components 3 \
    #      --window-size 30 \
    #      --step-size 5
    #
    # 2. Finally, run 'explore' on the new file:
    #    python v3_geometer.py explore \
    #      --input-embedding-file my_temporal_embedding.csv \
    #      --min-cluster-size 10 \
    #      --plot-prefix my_3d_plot
    
    main()

