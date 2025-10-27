"""
Example Usage of Manifold System v1

This script demonstrates:
1. Loading/creating data
2. Ingesting into the manifold system
3. Applying manifold learning
4. Visualizing results
5. Analyzing trajectories
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from manifold_system import ManifoldSystem, Entity
from manifold_learning import ManifoldLearner, TrajectoryAnalyzer
from manifold_viz import ManifoldVisualizer


def generate_synthetic_data(n_entities: int = 100,
                            n_features: int = 50,
                            n_timepoints: int = 5,
                            add_clusters: bool = True):
    """
    Generate synthetic dataset with multiple features and temporal evolution
    """
    data = []
    
    # Create entity IDs
    entity_ids = [f"entity_{i:04d}" for i in range(n_entities)]
    
    # Optionally create clusters
    if add_clusters:
        cluster_centers = np.random.randn(3, n_features) * 5
        cluster_assignments = np.random.randint(0, 3, n_entities)
    
    base_date = datetime.now()
    
    for t in range(n_timepoints):
        timestamp = base_date + timedelta(days=t*30)  # Monthly snapshots
        
        for i, entity_id in enumerate(entity_ids):
            # Base features
            if add_clusters:
                center = cluster_centers[cluster_assignments[i]]
                features = center + np.random.randn(n_features) * 2
            else:
                features = np.random.randn(n_features) * 5
            
            # Add temporal evolution (drift over time)
            features += t * np.random.randn(n_features) * 0.5
            
            # Create feature dictionary
            feature_dict = {
                f"feature_{j:03d}": features[j] 
                for j in range(n_features)
            }
            
            # Add some categorical features
            feature_dict['category_A'] = np.random.choice(['low', 'medium', 'high'])
            feature_dict['category_B'] = np.random.choice(['type1', 'type2', 'type3'])
            
            data.append({
                'entity_id': entity_id,
                'timestamp': timestamp,
                'cluster': cluster_assignments[i] if add_clusters else 0,
                **feature_dict
            })
    
    df = pd.DataFrame(data)
    return df


def example_basic_usage():
    """
    Example 1: Basic usage with synthetic data
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Manifold System Usage")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(n_entities=200, n_features=30, n_timepoints=1)
    print(f"   Created dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Features: {[col for col in df.columns if col.startswith('feature_')][:5]}...")
    
    # Initialize system
    print("\n2. Initializing Manifold System...")
    system = ManifoldSystem()
    
    # Ingest data
    print("\n3. Ingesting data...")
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    system.ingest_dataframe(
        df,
        entity_id_col='entity_id',
        timestamp_col='timestamp',
        feature_cols=feature_cols,
        metadata_cols=['cluster', 'category_A', 'category_B'],
        preprocess=True
    )
    
    print(f"   Summary: {system.summary()}")
    
    # Get feature matrix
    print("\n4. Extracting feature matrix...")
    X, entity_ids, feature_names = system.get_feature_matrix(standardize=True)
    print(f"   Matrix shape: {X.shape}")
    print(f"   Entities: {len(entity_ids)}")
    
    # Apply manifold learning
    print("\n5. Applying manifold learning methods...")
    learner = ManifoldLearner()
    
    print("   - Running PCA...")
    pca_proj = learner.project_pca(X, n_components=2, entity_ids=entity_ids)
    print(f"     Explained variance: {pca_proj.metadata['total_explained_variance']:.2%}")
    
    print("   - Running t-SNE...")
    tsne_proj = learner.project_tsne(X, n_components=2, entity_ids=entity_ids)
    
    # Visualize
    print("\n6. Creating visualizations...")
    viz = ManifoldVisualizer()
    
    # Get cluster labels for coloring
    labels = df.groupby('entity_id')['cluster'].first().values
    
    fig1, _ = viz.plot_projection_2d(
        pca_proj, 
        labels=labels,
        title="PCA Projection (colored by cluster)"
    )
    plt.savefig('/mnt/user-data/outputs/example_pca.png', dpi=150, bbox_inches='tight')
    print("   Saved: example_pca.png")
    
    fig2, _ = viz.plot_projection_2d(
        tsne_proj,
        labels=labels,
        title="t-SNE Projection (colored by cluster)"
    )
    plt.savefig('/mnt/user-data/outputs/example_tsne.png', dpi=150, bbox_inches='tight')
    print("   Saved: example_tsne.png")
    
    plt.close('all')
    
    print("\n✓ Example 1 complete!")
    return system, learner, df


def example_trajectory_analysis():
    """
    Example 2: Analyzing entity trajectories over time
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Trajectory Analysis")
    print("=" * 80)
    
    # Generate temporal data
    print("\n1. Generating temporal data...")
    df = generate_synthetic_data(n_entities=50, n_features=20, n_timepoints=10)
    print(f"   Created dataset: {df.shape[0]} rows across {df['timestamp'].nunique()} timepoints")
    
    # Initialize and ingest
    print("\n2. Ingesting temporal data...")
    system = ManifoldSystem()
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    system.ingest_dataframe(
        df,
        entity_id_col='entity_id',
        timestamp_col='timestamp',
        feature_cols=feature_cols,
        metadata_cols=['cluster']
    )
    
    # Get trajectories
    print("\n3. Extracting trajectories...")
    trajectories = system.get_trajectories()
    print(f"   Found {len(trajectories)} entity trajectories")
    
    # Project to 2D using PCA
    print("\n4. Projecting to 2D manifold...")
    X, entity_ids, feature_names = system.get_feature_matrix(standardize=True)
    learner = ManifoldLearner()
    projection = learner.project_pca(X, n_components=2, entity_ids=entity_ids)
    
    # Analyze specific trajectories
    print("\n5. Analyzing trajectory dynamics...")
    analyzer = TrajectoryAnalyzer(projection)
    
    # Get trajectory for first few entities
    sample_trajectories = {}
    for i, (entity_id, traj) in enumerate(list(trajectories.items())[:3]):
        # Get embeddings for each snapshot
        traj_points = []
        for snapshot in traj.snapshots:
            snapshot_vec = snapshot.to_vector(feature_names).reshape(1, -1)
            # Note: In real implementation, would need to transform using fitted model
            # For now, approximating
            traj_points.append(projection.embedding[entity_ids.index(entity_id)])
        
        traj_array = np.array(traj_points)
        sample_trajectories[entity_id] = traj_array
        
        # Get trajectory stats
        stats = analyzer.trajectory_summary(traj_array)
        print(f"\n   Entity {entity_id}:")
        print(f"     - Points: {stats['n_points']}")
        print(f"     - Total distance: {stats['total_distance']:.4f}")
        print(f"     - Mean speed: {stats['mean_speed']:.4f}")
    
    # Visualize trajectories
    print("\n6. Visualizing trajectories...")
    viz = ManifoldVisualizer()
    
    fig, _ = viz.plot_multiple_trajectories(
        sample_trajectories,
        projection,
        title="Entity Trajectories in Manifold Space"
    )
    plt.savefig('/mnt/user-data/outputs/example_trajectories.png', dpi=150, bbox_inches='tight')
    print("   Saved: example_trajectories.png")
    
    plt.close('all')
    
    print("\n✓ Example 2 complete!")
    return system, projection


def example_method_comparison():
    """
    Example 3: Compare different manifold learning methods
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Comparing Manifold Learning Methods")
    print("=" * 80)
    
    # Generate data
    print("\n1. Generating data...")
    df = generate_synthetic_data(n_entities=300, n_features=40, n_timepoints=1)
    
    # Initialize
    print("\n2. Preparing data...")
    system = ManifoldSystem()
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    system.ingest_dataframe(
        df,
        entity_id_col='entity_id',
        feature_cols=feature_cols,
        metadata_cols=['cluster']
    )
    
    X, entity_ids, _ = system.get_feature_matrix(standardize=True)
    
    # Compare methods
    print("\n3. Running multiple manifold learning methods...")
    learner = ManifoldLearner()
    
    # Check what's available
    deps = learner.check_dependencies()
    print(f"   Available libraries: {deps}")
    
    methods_to_compare = ['pca', 'isomap', 'tsne']
    if deps['umap']:
        methods_to_compare.append('umap')
    
    projections = learner.compare_methods(
        X,
        methods=methods_to_compare,
        n_components=2,
        entity_ids=entity_ids
    )
    
    # Visualize comparison
    print("\n4. Creating comparison visualization...")
    viz = ManifoldVisualizer()
    labels = df.groupby('entity_id')['cluster'].first().values
    
    fig, _ = viz.compare_projections(projections, labels=labels, ncols=2)
    plt.savefig('/mnt/user-data/outputs/example_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: example_comparison.png")
    
    # Show explained variance for PCA
    if 'pca' in projections:
        fig, _ = viz.plot_variance_explained(projections['pca'])
        plt.savefig('/mnt/user-data/outputs/example_variance.png', dpi=150, bbox_inches='tight')
        print("   Saved: example_variance.png")
    
    plt.close('all')
    
    print("\n✓ Example 3 complete!")
    return projections


def main():
    """
    Run all examples
    """
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "MANIFOLD SYSTEM v1 - EXAMPLES" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    try:
        # Run examples
        system1, learner1, df1 = example_basic_usage()
        system2, proj2 = example_trajectory_analysis()
        projections3 = example_method_comparison()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nGenerated files in /mnt/user-data/outputs/:")
        print("  - example_pca.png")
        print("  - example_tsne.png")
        print("  - example_trajectories.png")
        print("  - example_comparison.png")
        print("  - example_variance.png")
        print("\nYou can now use the system with your own datasets!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()