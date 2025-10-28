# Manifold System v1 - Interactive Edition 🎨

## What You Have Now

A **complete interactive manifold learning system** with three ways to use it!

## 📦 Complete Package

### Core System Files
- `manifold_system.py` - Entity management, data loading
- `manifold_learning.py` - All manifold algorithms
- `manifold_viz.py` - Static matplotlib visualizations

### Interactive Components (NEW!)
- `interactive_cli.py` - Menu-driven command-line interface
- `interactive_plotly.py` - Interactive Plotly visualizations
- `interactive_notebook.ipynb` - Jupyter notebook workflow
- `demo_interactive.py` - Quick demo script

### Demo Files (NEW!)
- `demo_pca_interactive.html` - Interactive PCA visualization
- `demo_umap_interactive.html` - Interactive UMAP visualization  
- `demo_comparison_interactive.html` - Side-by-side comparison

### Documentation
- `README.md` - Full system documentation
- `QUICKSTART.md` - 3-minute getting started
- `INTERACTIVE_GUIDE.md` - Guide to all interactive modes
- `TROUBLESHOOTING_WINDOWS.md` - Windows-specific help

### Setup & Utilities
- `requirements.txt` - All dependencies
- `setup_windows.ps1` - Windows PowerShell setup
- `setup_windows.bat` - Windows batch setup
- `quick_fix.py` - Fix common issues
- `debug_imports.py` - Debug tool
- `test_imports.py` - Test dependencies

## 🚀 Three Ways to Use It

### 1. Interactive CLI (No Coding!)

```bash
python interactive_cli.py
```

**Perfect for**:
- Quick exploration
- Learning the system
- Testing different algorithms
- No Python knowledge needed

**Features**:
- Load CSV or synthetic data
- Menu-driven workflow
- Run PCA, t-SNE, UMAP, Isomap, LLE, MDS
- Visualize results
- Export to CSV
- Compare methods side-by-side

### 2. Jupyter Notebook (Best Experience)

```bash
# Install first (if needed)
pip install jupyter plotly

# Launch
jupyter notebook interactive_notebook.ipynb
```

**Perfect for**:
- Data scientists
- Iterative exploration
- Documentation
- Beautiful visualizations

**Features**:
- Step-by-step workflow
- Inline static plots
- Interactive Plotly charts (zoomable, hoverable!)
- Modify parameters on the fly
- Export results
- Save your work

### 3. Python Scripts with Plotly

```python
from manifold_system import ManifoldSystem
from manifold_learning import ManifoldLearner
from interactive_plotly import InteractivePlotter

# Your workflow here...
# Creates beautiful interactive HTML plots!
```

**Perfect for**:
- Custom workflows
- Automation
- Integration with other tools
- Shareable visualizations

## ✨ Interactive Features

### What Makes It Interactive?

**Plotly Visualizations**:
- 🔍 **Zoom & Pan**: Click and drag to explore
- 👆 **Hover**: See entity details on mouseover
- 🎨 **Click Legend**: Toggle clusters on/off
- 💾 **Save**: Export as PNG or HTML
- 🔄 **Rotate**: 3D plots fully rotatable

**Example Hover Info**:
```
Entity: company_0123
Component 1: -0.234
Component 2: 1.567
Label: Cluster 2
```

### Live Demo Files

Open these in your browser to see the interactive features:

1. **demo_pca_interactive.html** - PCA projection
2. **demo_umap_interactive.html** - UMAP projection
3. **demo_comparison_interactive.html** - Compare methods

Just double-click the HTML files! They work offline.

## 🎯 Quick Start (Choose Your Adventure)

### Path 1: Try the Demo (2 minutes)

```bash
python demo_interactive.py
```

Opens 3 HTML files showing interactive visualizations. Double-click them!

### Path 2: Interactive CLI (5 minutes)

```bash
python interactive_cli.py

# Follow prompts:
1. Load Data → [2] Synthetic
2. Prepare Features → [2] Standardize
3. Run Projection → [6] UMAP
4. Visualize → [2] Colored by labels
```

### Path 3: Jupyter Notebook (10 minutes)

```bash
jupyter notebook interactive_notebook.ipynb

# Run cells in order
# Creates interactive plots in notebook
# Experiment with parameters
```

### Path 4: Use Your Own Data (15 minutes)

**Option A - CLI:**
```bash
python interactive_cli.py
[1] Load Data
  → [1] Load CSV file
  → path/to/your_data.csv
  → Follow prompts to select columns
```

**Option B - Jupyter:**
```python
# In notebook cell:
df = pd.read_csv('your_data.csv')

# Configure:
ENTITY_ID_COL = 'company_id'
FEATURE_COLS = ['revenue', 'employees', ...]

# Run rest of notebook
```

**Option C - Python Script:**
```python
from manifold_system import ManifoldSystem
from manifold_learning import ManifoldLearner
from interactive_plotly import InteractivePlotter
import pandas as pd

df = pd.read_csv('your_data.csv')

system = ManifoldSystem()
system.ingest_dataframe(df, entity_id_col='id', feature_cols=['f1', 'f2'])

X, ids, _ = system.get_feature_matrix(standardize=True)

learner = ManifoldLearner()
proj = learner.project_umap(X, entity_ids=ids)

plotter = InteractivePlotter()
fig = plotter.plot_projection_2d(proj)
fig.show()  # Opens in browser!
```

## 📊 Feature Comparison

| Feature | CLI | Jupyter | Python Script |
|---------|-----|---------|---------------|
| No coding | ✅ | ❌ | ❌ |
| Interactive plots | ❌* | ✅ | ✅ |
| Zoomable | ❌ | ✅ | ✅ |
| Menu-driven | ✅ | ❌ | ❌ |
| Customizable | ❌ | ✅ | ✅ |
| Save HTML | ❌ | ✅ | ✅ |
| Best for | Beginners | Data Scientists | Automation |

*CLI uses matplotlib (static) not Plotly (interactive)

## 🎨 What You Can Do

### Basic Operations
- ✅ Load CSV, JSON, or Parquet files
- ✅ Generate synthetic test data
- ✅ Handle 50-1000+ features per entity
- ✅ Mix continuous and categorical features
- ✅ Track entities over time (temporal data)

### Manifold Algorithms
- ✅ PCA (fast, linear)
- ✅ t-SNE (great for visualization)
- ✅ UMAP (best all-around)
- ✅ Isomap (geodesic distances)
- ✅ LLE (local structure)
- ✅ MDS (pairwise distances)

### Visualizations
- ✅ 2D and 3D projections
- ✅ Colored by labels/clusters
- ✅ Entity ID labels
- ✅ Trajectory paths (temporal)
- ✅ Side-by-side comparisons
- ✅ Variance explained plots
- ✅ Interactive HTML exports

### Analysis
- ✅ Identify clusters
- ✅ Find outliers
- ✅ Track trajectories
- ✅ Compare methods
- ✅ Export entities in regions
- ✅ Hover for details

## 💾 Installation

### Minimum (CLI + static plots)
```bash
pip install numpy pandas matplotlib scikit-learn umap-learn
```

### Full Interactive
```bash
pip install numpy pandas matplotlib scikit-learn umap-learn plotly jupyter
```

### Or use requirements.txt
```bash
pip install -r requirements.txt
```

## 🔧 Setup Scripts

### Windows PowerShell
```powershell
.\setup_windows.ps1
```

### Windows Command Prompt
```batch
setup_windows.bat
```

### Manual Setup
```bash
pip install -r requirements.txt
python quick_fix.py  # If issues
python demo_interactive.py  # Test it
```

## 🐛 Troubleshooting

### Import errors?
```bash
python quick_fix.py
```

### Plotly not working?
```bash
pip install plotly
python -c "import plotly; print('OK')"
```

### Jupyter not opening?
```bash
pip install jupyter notebook
jupyter notebook
```

### CLI plot window not showing?
- Try Jupyter instead (better for remote)
- Check matplotlib backend
- Use Plotly for guaranteed browser-based plots

## 📈 Example Use Cases

### Use Case 1: Explore Company Data
```python
# 200 companies, 50 financial metrics
# Find clusters of similar companies
# Identify outliers
# Track over quarters
```

### Use Case 2: Patient Clustering
```python
# 1000 patients, 100 biomarkers
# Identify patient subgroups
# Predict trajectories
# Find similar patients
```

### Use Case 3: Product Intelligence
```python
# 500 products, 200 features
# Product recommendations
# Market positioning
# Trend analysis
```

## 🎯 Next Steps

### Immediate
1. ✅ Run `demo_interactive.py` to see it working
2. ✅ Try `interactive_cli.py` for menu-driven use
3. ✅ Open `interactive_notebook.ipynb` in Jupyter

### This Week
1. Load your actual dataset
2. Try different manifold algorithms
3. Find optimal parameters
4. Export results

### This Month  
1. Integrate into your workflow
2. Build custom analysis scripts
3. Share interactive HTML visualizations
4. Prepare for v2 (production scale)

## 🌟 Tips for Success

1. **Start simple**: Use synthetic data first
2. **Compare methods**: Different algorithms for different data
3. **Iterate quickly**: Use Jupyter for experimentation
4. **Save HTML**: Share interactive plots with team
5. **Document findings**: Notebook keeps your work
6. **Export early**: Save projections as CSV
7. **Use labels**: Color by meaningful categories
8. **Hover to explore**: Interactive plots reveal insights
9. **Zoom in**: Look for sub-clusters
10. **Trust the process**: Manifold learning reveals hidden structure

## 📚 Documentation Quick Links

- **Get Started**: See `QUICKSTART.md`
- **Full Docs**: See `README.md`
- **Interactive Guide**: See `INTERACTIVE_GUIDE.md`
- **Windows Help**: See `TROUBLESHOOTING_WINDOWS.md`

## 🎉 You're Ready!

You now have:
- ✅ Complete manifold learning system
- ✅ Three interactive interfaces
- ✅ Beautiful visualizations
- ✅ Working examples
- ✅ Full documentation

Choose your interface and start exploring your data in geometric space! 🚀

---

## Quick Command Reference

```bash
# Try the demo
python demo_interactive.py

# Interactive CLI
python interactive_cli.py

# Jupyter notebook
jupyter notebook interactive_notebook.ipynb

# Test your setup
python quick_fix.py

# Debug issues
python debug_imports.py

# Static examples
python example_usage.py
```

**Questions?** Check the documentation files or run the debug scripts.

**Ready to build Manifold!** 🏔️
