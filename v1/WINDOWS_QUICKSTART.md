# Windows Quick Start Guide

## ✅ Fixed! All scripts now work on Windows

The system has been updated to work cross-platform. Use these new scripts:

## 🚀 Quick Commands (Windows)

### 1. Simple Demo (Recommended First)
```powershell
python demo_simple.py
```
This creates an `outputs` folder in your current directory with the visualizations.

### 2. Full Examples
```powershell
python example_usage_crossplatform.py
```

### 3. Interactive CLI
```powershell
python interactive_cli.py
```

### 4. Jupyter Notebook
```powershell
jupyter notebook interactive_notebook.ipynb
```

## 📁 Where Are My Files?

All outputs are now saved to `./outputs/` in your current directory.

On Windows, this will be something like:
```
C:\Users\theoe\OneDrive\Documents\GitHub\manifold\v1\outputs\
```

## 🔧 Setup (If Needed)

### Install Dependencies
```powershell
pip install numpy pandas matplotlib scikit-learn umap-learn
```

### For Interactive Features
```powershell
pip install plotly jupyter
```

### Or Install Everything
```powershell
pip install -r requirements.txt
```

## ⚡ Troubleshooting

### Issue: "No such file or directory: '/mnt/user-data/outputs/...'"

**Solution**: Use the new cross-platform scripts:
- ❌ `example_usage.py` (old, Linux-only)
- ✅ `example_usage_crossplatform.py` (new, works everywhere)
- ✅ `demo_simple.py` (new, works everywhere)

### Issue: Import errors

**Solution**:
```powershell
python quick_fix.py
```

### Issue: "Jupyter not found"

**Solution**:
```powershell
pip install jupyter notebook
```

## 📂 Updated File List

### Use These (Cross-Platform):
- ✅ `demo_simple.py` - Quick demo
- ✅ `example_usage_crossplatform.py` - Full examples
- ✅ `interactive_cli.py` - Menu interface
- ✅ `interactive_notebook.ipynb` - Jupyter notebook
- ✅ `path_utils.py` - Path handling utility

### Avoid These on Windows:
- ❌ `example_usage.py` (has hardcoded Linux paths)
- ❌ `demo_interactive.py` (has hardcoded Linux paths)

## 🎯 Recommended Workflow (Windows)

1. **First Time**:
   ```powershell
   # Install dependencies
   pip install -r requirements.txt
   
   # Run simple demo
   python demo_simple.py
   
   # Check outputs folder
   dir outputs
   ```

2. **Explore with CLI**:
   ```powershell
   python interactive_cli.py
   ```
   Follow the menus to load data and create projections.

3. **Use Jupyter** (best experience):
   ```powershell
   jupyter notebook interactive_notebook.ipynb
   ```
   Run cells in order for interactive plots.

4. **Your Own Scripts**:
   ```python
   from pathlib import Path
   from manifold_system import ManifoldSystem
   from manifold_learning import ManifoldLearner
   
   # Create outputs directory
   OUTPUT_DIR = Path('./outputs')
   OUTPUT_DIR.mkdir(exist_ok=True)
   
   # Your code here...
   
   # Save with Path
   output_file = OUTPUT_DIR / 'my_plot.png'
   plt.savefig(str(output_file))
   ```

## 🌟 Key Differences from Linux Version

| Item | Linux | Windows |
|------|-------|---------|
| Output path | `/mnt/user-data/outputs/` | `./outputs/` |
| Path separator | `/` | `\` or `/` (both work) |
| Use Path objects | Optional | **Recommended** |

## 💡 Tips

1. **Always use Path objects** for cross-platform compatibility:
   ```python
   from pathlib import Path
   output = Path('./outputs') / 'myfile.png'
   ```

2. **Convert Path to string** when saving:
   ```python
   plt.savefig(str(output))
   fig.write_html(str(output))
   ```

3. **Check where you are**:
   ```python
   import os
   print(f"Current directory: {os.getcwd()}")
   ```

4. **Use relative paths** not absolute:
   - ✅ `./outputs/file.png`
   - ❌ `C:\Users\...\outputs\file.png`

## 📊 Example Session

```powershell
PS C:\Users\theoe\OneDrive\Documents\GitHub\manifold\v1> python demo_simple.py

======================================================================
MANIFOLD SYSTEM - QUICK DEMO (Cross-Platform)
======================================================================

Output directory: C:\Users\theoe\OneDrive\Documents\GitHub\manifold\v1\outputs

Step 1: Generating synthetic data...
  Created 200 entities with 30 features

Step 2: Setting up manifold system...
  Feature matrix: (200, 30)

Step 3: Running manifold projections...
  - PCA... ✓ (75.2% variance)
  - UMAP... ✓

Step 4: Creating visualizations...
  ✓ Saved: demo_pca.html (interactive)
  ✓ Saved: demo_umap.html (interactive)

======================================================================
SUCCESS! 🎉
======================================================================

Open the HTML files in outputs to explore!
You can zoom, pan, and hover over points.

PS C:\Users\theoe\OneDrive\Documents\GitHub\manifold\v1> dir outputs


    Directory: C:\Users\theoe\OneDrive\Documents\GitHub\manifold\v1\outputs


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        10/27/2025   2:30 PM        4823456 demo_pca.html
-a----        10/27/2025   2:30 PM        4823456 demo_umap.html
```

## ✅ Verification

After running `demo_simple.py`, you should see:
1. An `outputs` folder created in your current directory
2. HTML or PNG files inside the outputs folder
3. No errors about missing directories

If you see these, **you're all set!** 🎉

## 🆘 Still Having Issues?

Run the diagnostic:
```powershell
python debug_imports.py
```

Or check the full troubleshooting guide:
```powershell
notepad TROUBLESHOOTING_WINDOWS.md
```

---

**You're ready to use the Manifold system on Windows!** 🚀
