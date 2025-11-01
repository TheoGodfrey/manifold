Project: Manifold - A Data-to-Discovery PipelineThis project is an end-to-end pipeline to ingest raw, noisy, time-series data, clean it using hierarchical Bayesian models, and then discover hidden "geometric laws" and "trajectories" using manifold learning.Project ArchitectureThe pipeline is broken into three main stages:Stage 1: V1 - The Data Foundation (Real-time)Purpose: To capture and store all raw, arbitrary data streams 24/7.Scripts:schema_registry.py: Defines and validates the structure of incoming data.ingestion_api.py: A Flask API that receives data, validates it, and sends it to Kafka.storage_writer.py: A Kafka consumer that reads the data stream and writes it to a partitioned Parquet data lake (manifold_data_lake/).Output: A data lake full of noisy, sparse, but well-structured Parquet files.Stage 2: V2 - The Bayesian Estimator (Offline, Batch)Purpose: To clean the raw V1 data. It de-noises, "fills in" missing values, and calculates the "True Value" for a field by pooling information from all entities.Script: hierarchical_bayesian_model.pyWorkflow:First, you must manually run an ETL job (not included) to query all V1 Parquet files (e.g., for plasma_temp) and save them as a single, long-format CSV with NaN for missing data.Run the hierarchical_bayesian_model.py run command on this CSV.Output: A single, clean, complete CSV (e.g., hbm_estimates.csv) containing the true_value_mean and confidence intervals for every entity at every time step.Stage 3: V3/V4 - The Geometer (Offline, Analytics)Purpose: To analyze the clean V2 data to find "geometric laws" (states) and "trajectories" (evolution over time).Script: v3_geometer.pyWorkflow:v3_geometer.py run: Takes the V2 hbm_estimates.csv as input. It uses a sliding window to create temporal vectors, then runs UMAP to generate a 3D (or 2D) embedding.v3_geometer.py explore: Takes the embedding file from the run step. It runs HDBSCAN to find clusters ("states") and generates two interactive 3D plots.How to Run the Full Pipeline (Example)Prerequisites:Python 3.10+Kafka and Zookeeper running (e.g., via Docker)Required Python packages: pip install flask kafka-python pyarrow pymc numpy pandas matplotlib arviz umap-learn hdbscan plotly scikit-learnStep 0: Start V1 ServicesIn separate terminals, start the V1 services:# Terminal 1: Start the API
python ingestion_api.py

# Terminal 2: Start the Storage Writer
python storage_writer.py
Now, your pipeline is "live" and collecting data in manifold_data_lake/. Let this run for a while.Step 1: Run V2 Estimator (e.g., nightly)Run your ETL job (Manual): Query the manifold_data_lake/ and create a single CSV file, e.g., all_plasma_data.csv. (For testing, you can use the make-dummy-data command).# (One-time only) Create dummy data for testing:
python hierarchical_bayesian_model.py make-dummy-data --file-name dummy_data.csv
Run the V2 analysis:python hierarchical_bayesian_model.py run \
  --data-file dummy_data.csv \
  --entity-col reactor_id \
  --time-col day \
  --field-col plasma_temp \
  --output-file hbm_estimates.csv \
  --plot-prefix v2_plot
Output: hbm_estimates.csv (the clean data) and v2_plot_*.png (diagnostic plots).Step 2: Run V3/V4 Geometer (Analytics)Run the run command to generate the 3D temporal embedding.python v3_geometer.py run \
  --input-file hbm_estimates.csv \
  --entity-col entity \
  --time-col time \
  --field-col true_value_mean \
  --output-embedding-file v4_embedding.csv \
  --n-components 3 \
  --window-size 30 \
  --step-size 5
Output: v4_embedding.csv (the 3D coordinates).Run the explore command to cluster and visualize the embedding.python v3_geometer.py explore \
  --input-embedding-file v4_embedding.csv \
  --min-cluster-size 10 \
  --plot-prefix v4_plot
Output: v4_plot_states.html and v4_plot_trajectories.html.You can now open the .html files in your browser to interact with your 3D manifold.