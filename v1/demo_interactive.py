"""
Quick Interactive Demo
Shows off the interactive Plotly visualizations
"""

import numpy as np
import pandas as pd
from manifold_system import ManifoldSystem
from manifold_learning import ManifoldLearner
from example_usage import generate_synthetic_data

print("=" * 70)
print("MANIFOLD SYSTEM - INTERACTIVE DEMO")
print("=" * 70)
print()

# Check if plotly available
try:
    from interactive_plotly import InteractivePlotter
    print("✓ Plotly available - Interactive mode enabled!")
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠ Plotly not available. Install with: pip install plotly")
    print("  Falling back to matplotlib...")
    PLOTLY_AVAILABLE = False
    from manifold_viz import ManifoldVisualizer

print()

# Generate data
print("Generating synthetic data...")
df = generate_synthetic_data(n_entities=200, n_features=30, n_timepoints=1, add_clusters=True)
print(f"✓ Created {len(df)} entities with 30 features")
print()

# Setup system
print("Initializing system...")
system = ManifoldSystem()
feature_cols = [c for c in df.columns if c.startswith('feature_')]

system.ingest_dataframe(
    df,
    entity_id_col='entity_id',
    timestamp_col='timestamp',
    feature_cols=feature_cols,
    metadata_cols=['cluster']
)

X, entity_ids, feature_names = system.get_feature_matrix(standardize=True)
labels = df.groupby('entity_id')['cluster'].first().values

print(f"✓ Feature matrix: {X.shape}")
print()

# Run projections
print("Running manifold projections...")
learner = ManifoldLearner()

print("  - PCA...", end=" ", flush=True)
pca_proj = learner.project_pca(X, n_components=2, entity_ids=entity_ids)
print(f"✓ (variance: {pca_proj.metadata['total_explained_variance']:.2%})")

print("  - UMAP...", end=" ", flush=True)
umap_proj = learner.project_umap(X, n_components=2, entity_ids=entity_ids)
print("✓")

print()

# Visualize
if PLOTLY_AVAILABLE:
    print("Creating interactive visualizations...")
    plotter = InteractivePlotter()
    
    # PCA plot
    print("  - PCA projection (interactive)")
    fig = plotter.plot_projection_2d(
        pca_proj,
        labels=labels,
        title="PCA Projection - Interactive Demo"
    )
    
    # Save as HTML
    output_file = '/mnt/user-data/outputs/demo_pca_interactive.html'
    fig.write_html(output_file)
    print(f"    Saved to: demo_pca_interactive.html")
    
    # UMAP plot
    print("  - UMAP projection (interactive)")
    fig = plotter.plot_projection_2d(
        umap_proj,
        labels=labels,
        title="UMAP Projection - Interactive Demo"
    )
    
    output_file = '/mnt/user-data/outputs/demo_umap_interactive.html'
    fig.write_html(output_file)
    print(f"    Saved to: demo_umap_interactive.html")
    
    # Comparison
    print("  - Method comparison (interactive)")
    projections = {'PCA': pca_proj, 'UMAP': umap_proj}
    fig = plotter.compare_projections_interactive(projections, labels=labels)
    
    output_file = '/mnt/user-data/outputs/demo_comparison_interactive.html'
    fig.write_html(output_file)
    print(f"    Saved to: demo_comparison_interactive.html")
    
    print()
    print("=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print()
    print("Interactive HTML files created! Open them in your browser:")
    print("  1. demo_pca_interactive.html")
    print("  2. demo_umap_interactive.html")
    print("  3. demo_comparison_interactive.html")
    print()
    print("In these files you can:")
    print("  - Zoom and pan")
    print("  - Hover to see entity details")
    print("  - Click legend to toggle clusters")
    print("  - Save as image")
    
else:
    # Fallback to matplotlib
    print("Creating static visualizations...")
    import matplotlib.pyplot as plt
    viz = ManifoldVisualizer()
    
    fig, ax = viz.plot_projection_2d(pca_proj, labels=labels)
    plt.savefig('/mnt/user-data/outputs/demo_pca_static.png', dpi=150, bbox_inches='tight')
    print("  Saved: demo_pca_static.png")
    
    fig, ax = viz.plot_projection_2d(umap_proj, labels=labels)
    plt.savefig('/mnt/user-data/outputs/demo_umap_static.png', dpi=150, bbox_inches='tight')
    print("  Saved: demo_umap_static.png")
    
    plt.close('all')
    
    print()
    print("Created static plots. Install plotly for interactive features:")
    print("  pip install plotly")

print()
