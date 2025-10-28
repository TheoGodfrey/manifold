# Interactive Manifold System - User Guide

You now have **three interactive ways** to use the Manifold System!

## 🎯 Choose Your Interface

### 1. **Interactive CLI** (Command Line)
**Best for**: Quick exploration, terminal users, running on servers

**Start it**:
```bash
python interactive_cli.py
```

**Features**:
- Menu-driven interface
- Load CSV or generate synthetic data
- Run different manifold algorithms
- Visualize results
- Export projections
- No coding required!

**How it works**:
```
╔══════════════════════════════════════════════════════════╗
║          MANIFOLD SYSTEM v1 - INTERACTIVE                ║
╚══════════════════════════════════════════════════════════╝

Main Menu:
  [1] Load Data
  [2] Prepare Features  
  [3] Run Projection
  [4] Visualize
  [5] Compare Methods
  [6] Trajectory Analysis
  [7] Export Results
  [q] Quit

Choose an option: 
```

Just follow the prompts!

---

### 2. **Jupyter Notebook** (Interactive Coding)
**Best for**: Data scientists, iterative exploration, documentation

**Start it**:
```bash
jupyter notebook interactive_notebook.ipynb
```

**Features**:
- Step-by-step workflow
- Inline visualizations
- Modify parameters easily
- Save your work
- Interactive Plotly charts (zoomable, hoverable)

**Workflow**:
1. Load data
2. Configure parameters
3. Run projections
4. Visualize (static + interactive)
5. Analyze results
6. Export

**Example cell**:
```python
# Just run this cell to project with UMAP
umap_proj = learner.project_umap(
    X, 
    n_components=2,
    n_neighbors=15
)

# Interactive plot
fig = plotter.plot_projection_2d(umap_proj, labels=labels)
fig.show()  # Zoom, pan, hover to explore!
```

---

### 3. **Interactive Plotly Visualizations** (Web-based)
**Best for**: Beautiful, shareable visualizations

**Use it in your code**:
```python
from interactive_plotly import InteractivePlotter

plotter = InteractivePlotter()

# Create interactive 2D plot
fig = plotter.plot_projection_2d(
    projection,
    labels=labels
)

# Show in browser
fig.show()

# Or save as HTML
fig.write_html('my_projection.html')
# Open in browser for full interactivity!
```

**Features**:
- Zoom and pan
- Hover to see entity details
- Click to select points
- Rotate 3D plots
- Save as HTML (shareable!)

---

## 🚀 Quick Start Examples

### Example 1: CLI Mode (No coding)

```bash
python interactive_cli.py

# Then follow prompts:
[1] Load Data
  → [2] Load example synthetic data
  → 200 entities, 30 features, 1 timepoint

[2] Prepare Features
  → [2] Standardize (z-score)

[3] Run Projection
  → [6] UMAP
  → Components: 2
  → Neighbors: 15
  → Min distance: 0.1

[4] Visualize
  → [1] (select UMAP)
  → [2] Colored by labels

# Plot opens in window!
```

### Example 2: Jupyter Notebook

```bash
jupyter notebook interactive_notebook.ipynb

# Run cells in order:
# 1. Imports ✓
# 2. Initialize ✓
# 3. Load data (synthetic or your CSV)
# 4. Configure columns
# 5. Ingest data
# 6. Prepare features
# 7. Run PCA (fast)
# 8. Run UMAP (better)
# 9. See interactive plots!
```

### Example 3: Python Script with Interactive Plots

```python
from manifold_system import ManifoldSystem
from manifold_learning import ManifoldLearner
from interactive_plotly import InteractivePlotter
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')

# Setup
system = ManifoldSystem()
system.ingest_dataframe(df, entity_id_col='id', feature_cols=['f1', 'f2', 'f3'])

# Project
X, ids, features = system.get_feature_matrix(standardize=True)
learner = ManifoldLearner()
proj = learner.project_umap(X, n_components=2, entity_ids=ids)

# Interactive plot
plotter = InteractivePlotter()
fig = plotter.plot_projection_2d(proj)
fig.show()

# Save for sharing
fig.write_html('results.html')
```

---

## 📊 Interactive Features Comparison

| Feature | CLI | Jupyter | Plotly Script |
|---------|-----|---------|---------------|
| No coding needed | ✅ | ❌ | ❌ |
| Menu-driven | ✅ | ❌ | ❌ |
| Interactive viz | ✅* | ✅ | ✅ |
| Zoomable plots | ❌ | ✅ | ✅ |
| Hover data | ❌ | ✅ | ✅ |
| Save as HTML | ❌ | ✅ | ✅ |
| Step-by-step | ✅ | ✅ | ❌ |
| Customizable | ❌ | ✅ | ✅ |
| Remote server | ✅ | ⚠️ | ⚠️ |

*CLI shows matplotlib plots (not interactive Plotly)

---

## 🎨 Interactive Visualization Features

### Plotly Interactions

**In browser**:
- **Zoom**: Click and drag to zoom
- **Pan**: Hold shift + drag to pan
- **Hover**: See entity details
- **Select**: Click legend to toggle groups
- **Reset**: Double-click to reset view
- **Save**: Camera icon to save as PNG

**3D Plots** (if n_components=3):
- **Rotate**: Click and drag
- **Zoom**: Scroll
- **Pan**: Right-click and drag

### What You Can See on Hover

```
Entity: entity_0123
Component 1: -0.234
Component 2: 1.567
[Any custom data you add]
```

---

## 💡 Tips & Tricks

### CLI Mode Tips

1. **Start with synthetic data** to test the system
2. **Use Compare Methods** to see which algorithm works best
3. **Export results** before quitting
4. **Take notes** on which parameters work well

### Jupyter Tips

1. **Run cells in order** the first time
2. **Experiment with parameters** by re-running cells
3. **Use `%%time`** to see how long projections take
4. **Save the notebook** with your results
5. **Create new cells** to try different things

### Plotly Tips

1. **Save interactive HTML** to share with others
2. **Add custom hover data** for your entities
3. **Use 3D mode** for high-dimensional data
4. **Compare multiple projections** side-by-side

---

## 🔧 Installation Requirements

### Basic (CLI + Matplotlib viz)
```bash
pip install numpy pandas matplotlib scikit-learn umap-learn
```

### Interactive (Jupyter + Plotly)
```bash
pip install numpy pandas matplotlib scikit-learn umap-learn plotly jupyter
```

### Check if installed:
```python
python -c "import plotly; print('Plotly:', plotly.__version__)"
python -c "import jupyter; print('Jupyter: OK')"
```

---

## 📝 Example Workflows

### Workflow 1: Quick Data Exploration (5 min)

```bash
python interactive_cli.py

1. Load synthetic data (200 entities, 30 features)
2. Prepare features (standardize)
3. Run UMAP projection
4. Visualize with labels
5. Export to CSV
```

### Workflow 2: Compare Methods (10 min)

```bash
python interactive_cli.py

1. Load your CSV
2. Prepare features
3. Use "Compare Methods"
   - Runs PCA, t-SNE, UMAP, Isomap
   - Shows all side-by-side
4. Pick the best one
5. Export results
```

### Workflow 3: Interactive Analysis (30 min)

```bash
jupyter notebook interactive_notebook.ipynb

1. Load your dataset
2. Run all projection methods
3. Create interactive Plotly charts
4. Zoom into interesting regions
5. Identify outliers
6. Export entities of interest
7. Save HTML for sharing
```

---

## 🐛 Troubleshooting

### "Plotly not available"
```bash
pip install plotly
```

### "Jupyter not found"
```bash
pip install jupyter
```

### CLI plot window doesn't open
- Check if `matplotlib` is installed
- Try adding `%matplotlib inline` in Jupyter
- On remote servers, use Jupyter instead of CLI

### Interactive plot not showing in Jupyter
```python
# Make sure plotly is imported
import plotly.graph_objects as go

# Enable inline plotting
from IPython.display import display
```

---

## 🎯 Which Mode Should I Use?

**Use CLI if you**:
- Want a quick demo
- Don't want to code
- Are on a server with no GUI
- Just want to try different algorithms

**Use Jupyter if you**:
- Want to explore interactively
- Need to tweak parameters
- Want to document your work
- Want beautiful interactive plots

**Use Python scripts with Plotly if you**:
- Have a specific workflow
- Want to automate
- Need to integrate with other tools
- Want to share HTML visualizations

---

## 🌟 Next Steps

1. **Try the CLI** to get familiar: `python interactive_cli.py`
2. **Open the notebook** for deeper exploration: `jupyter notebook interactive_notebook.ipynb`
3. **Use Plotly** for final visualizations
4. **Build your own workflow** combining all three!

All three modes use the same underlying system, so you can switch between them as needed.

---

**Happy exploring! 🚀**
