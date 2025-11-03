#!/usr/bin/env python3
"""
Quick diagnostic script to identify processing_v2 issues
Run this first to see what's wrong!
"""

import sys
from pathlib import Path

print("="*70)
print("PROCESSING_V2 DIAGNOSTIC TOOL")
print("="*70)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 6):
    print("  ⚠️  WARNING: Python 3.6+ recommended")

# Check dependencies
print("\n📦 Checking dependencies...")
missing = []

try:
    import pandas
    print(f"  ✓ pandas {pandas.__version__}")
except ImportError:
    print("  ✗ pandas NOT INSTALLED")
    missing.append("pandas")

try:
    import tqdm
    print(f"  ✓ tqdm installed")
except ImportError:
    print("  ✗ tqdm NOT INSTALLED")
    missing.append("tqdm")

if missing:
    print(f"\n❌ Missing packages! Install with:")
    print(f"   pip install {' '.join(missing)}")
    print()
else:
    print("\n✅ All dependencies installed!")

# Check for ZIP files
print("\n📁 Checking for ZIP files in current directory...")
current_dir = Path.cwd()
print(f"   Current directory: {current_dir}")

zip_files = list(current_dir.glob("Accounts_Monthly_Data-April*.zip"))

if zip_files:
    print(f"\n✅ Found {len(zip_files)} ZIP files:")
    for zf in sorted(zip_files):
        size_mb = zf.stat().st_size / (1024*1024)
        print(f"   • {zf.name} ({size_mb:.1f} MB)")
else:
    print("\n❌ No ZIP files found in current directory!")
    print("\n   Looking for files like: Accounts_Monthly_Data-April2010.zip")
    print("\n   Solutions:")
    print("   1. Move ZIP files to this directory")
    print("   2. Run this script from the directory with ZIP files")
    print("   3. Edit DATA_DIR in processing_v2_fixed.py")

# Check for checkpoints
print("\n💾 Checking for checkpoint files...")
checkpoints = list(current_dir.glob("checkpoint_*.pkl"))
if checkpoints:
    print(f"   Found {len(checkpoints)} checkpoint files")
    for cp in sorted(checkpoints):
        print(f"   • {cp.name}")
else:
    print("   No checkpoint files found (this is normal for first run)")

# Check disk space
try:
    import shutil
    total, used, free = shutil.disk_usage(current_dir)
    free_gb = free / (1024**3)
    print(f"\n💽 Disk space: {free_gb:.1f} GB free")
    if free_gb < 10:
        print("   ⚠️  WARNING: Low disk space! Consider freeing up space.")
except:
    pass

# Final recommendation
print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)

if missing:
    print(f"1. Install missing packages: pip install {' '.join(missing)}")
    print("2. Re-run this diagnostic")
elif not zip_files:
    print("1. Place your ZIP files in:", current_dir)
    print("2. Re-run this diagnostic")
    print("3. Once ZIP files are found, run: python processing_v2_fixed.py")
else:
    print("✅ Everything looks good!")
    print("   Run: python processing_v2_fixed.py")
    print()
    print("   (It will start in TEST_MODE by default)")

print("="*70 + "\n")
