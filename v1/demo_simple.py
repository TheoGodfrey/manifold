"""
Quick Demo - Cross-Platform Version
Works on Windows, Mac, and Linux!
"""

import numpy as np
import pandas as pd
from pathlib import Path
from manifold_system import ManifoldSystem
from manifold_learning import ManifoldLearner

# Create outputs directory
OUTPUT_DIR = Path('./outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("MANIFOLD SYSTEM - QUICK DEMO (Cross-Platform)")
print("=" * 70)
print()
print(f"Output directory: {OUTPUT_DIR.absolute()}")
print()

# Generate simple synthetic data
def generate_data(n=200, n_features=30):
    """Generate simple test data with 3 clusters"""
    from datetime import datetime
    
    cluster_centers = np.random.randn(3, n_features) * 5
    clusters = np.random.randint(0, 3, n)
    
    data = []
    for i in range(n):
        features = cluster_centers[clusters[i]] + np.random.randn(n_features) * 2
        row = {
            'entity_id': f'entity_{i:04d}',
            'timestamp': datetime.now(),
            'cluster': clusters[i]
        }
        row.update({f'feature_{j:03d}': features[j] for j in range(n_features)})
        data.append(row)
    
    return pd.DataFrame(data)

# Step 1: Generate data
print("Step 1: Generating synthetic data...")
df = generate_data()
print(f"  Created {len(df)} entities with 30 features")
print()

# Step 2: Setup system
print("Step 2: Setting up manifold system...")
system = ManifoldSystem()
feature_cols = [c for c in df.columns if c.startswith('feature_')]

system.ingest_dataframe(
    df,
    entity_id_col='entity_id',
    timestamp_col='timestamp',
    feature_cols=feature_cols,
    metadata_cols=['cluster']
)

X, entity_ids, _ = system.get_feature_matrix(standardize=True)
labels = df['cluster'].values
print(f"  Feature matrix: {X.shape}")
print()

# Step 3: Run projections
print("Step 3: Running manifold projections...")
learner = ManifoldLearner()

print("  - PCA...", end=" ", flush=True)
pca_proj = learner.project_pca(X, n_components=2, entity_ids=entity_ids)
print(f"✓ ({pca_proj.metadata['total_explained_variance']:.1%} variance)")

print("  - UMAP...", end=" ", flush=True)
if learner.check_dependencies()['umap']:
    umap_proj = learner.project_umap(X, n_components=2, entity_ids=entity_ids)
    print("✓")
else:
    print("⚠ (not installed)")
    umap_proj = None
print()

# Step 4: Visualize
print("Step 4: Creating visualizations...")

# Try interactive plots first
try:
    from interactive_plotly import InteractivePlotter
    
    plotter = InteractivePlotter()
    
    # PCA
    fig = plotter.plot_projection_2d(pca_proj, labels=labels, title="PCA Demo")
    output_file = OUTPUT_DIR / 'demo_pca.html'
    fig.write_html(str(output_file))
    print(f"  ✓ Saved: {output_file.name} (interactive)")
    
    # UMAP
    if umap_proj:
        fig = plotter.plot_projection_2d(umap_proj, labels=labels, title="UMAP Demo")
        output_file = OUTPUT_DIR / 'demo_umap.html'
        fig.write_html(str(output_file))
        print(f"  ✓ Saved: {output_file.name} (interactive)")
    
    print()
    print("=" * 70)
    print("SUCCESS! 🎉")
    print("=" * 70)
    print()
    print(f"Open the HTML files in {OUTPUT_DIR} to explore!")
    print("You can zoom, pan, and hover over points.")
    
except ImportError:
    # Fallback to matplotlib
    print("  (Plotly not installed, using matplotlib)")
    import matplotlib.pyplot as plt
    from manifold_viz import ManifoldVisualizer
    
    viz = ManifoldVisualizer()
    
    fig, _ = viz.plot_projection_2d(pca_proj, labels=labels)
    output_file = OUTPUT_DIR / 'demo_pca.png'
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    
    if umap_proj:
        fig, _ = viz.plot_projection_2d(umap_proj, labels=labels)
        output_file = OUTPUT_DIR / 'demo_umap.png'
        plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")
    
    plt.close('all')
    
    print()
    print("=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print()
    print(f"Check {OUTPUT_DIR} for the generated plots.")
    print()
    print("Install plotly for interactive visualizations:")
    print("  pip install plotly")

print()
