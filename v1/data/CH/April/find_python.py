#!/usr/bin/env python3
"""
Find which Python interpreter has pandas and tqdm installed
"""

import sys
import subprocess
from pathlib import Path

print("="*70)
print("PYTHON INTERPRETER DIAGNOSTIC")
print("="*70)

# Show current Python
print(f"\n🐍 Currently using:")
print(f"   Python: {sys.executable}")
print(f"   Version: {sys.version.split()[0]}")

# Try to import packages with THIS Python
print(f"\n📦 Checking packages with THIS Python ({sys.executable}):")
packages = {}

for pkg in ['pandas', 'tqdm']:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"   ✓ {pkg} {version}")
        packages[pkg] = True
    except ImportError:
        print(f"   ✗ {pkg} NOT INSTALLED")
        packages[pkg] = False

# If all packages installed, we're good!
if all(packages.values()):
    print("\n" + "="*70)
    print("✅ SOLUTION FOUND!")
    print("="*70)
    print(f"\nUse this command to run your scripts:")
    print(f"   {sys.executable} processing_v2_fixed.py")
    print("\nOR create an alias:")
    print(f"   alias python='{sys.executable}'")
    sys.exit(0)

# Otherwise, check other Python interpreters
print("\n" + "="*70)
print("🔍 Searching for other Python interpreters...")
print("="*70)

# Common Python commands to try
python_commands = [
    'python', 'python3', 'py', 
    'python3.11', 'python3.10', 'python3.9', 'python3.8',
    'python2', 'python2.7'
]

found_solutions = []

for cmd in python_commands:
    try:
        # Get the path
        result = subprocess.run(
            [cmd, '-c', 'import sys; print(sys.executable)'],
            capture_output=True, text=True, timeout=2
        )
        
        if result.returncode != 0:
            continue
            
        exe_path = result.stdout.strip()
        
        # Skip if it's the same as current
        if exe_path == sys.executable:
            continue
        
        # Check for packages
        check_result = subprocess.run(
            [cmd, '-c', 'import pandas, tqdm; print("OK")'],
            capture_output=True, text=True, timeout=2
        )
        
        if check_result.returncode == 0 and "OK" in check_result.stdout:
            found_solutions.append((cmd, exe_path))
            print(f"\n✅ Found working Python:")
            print(f"   Command: {cmd}")
            print(f"   Path: {exe_path}")
            print(f"   Has pandas & tqdm: YES")
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        continue

# Summary and recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)

if found_solutions:
    print(f"\n✅ Found {len(found_solutions)} working Python interpreter(s)!\n")
    
    for i, (cmd, path) in enumerate(found_solutions, 1):
        print(f"Option {i}: Use '{cmd}'")
        print(f"   Run: {cmd} processing_v2_fixed.py")
        print()
    
    print("Pick one and use that command from now on!")
    
else:
    print("\n❌ No Python found with both pandas and tqdm installed.\n")
    print("SOLUTION: Install to the Python you're using:")
    print(f"   {sys.executable} -m pip install pandas tqdm")
    print("\nThen verify:")
    print(f"   {sys.executable} diagnose.py")

print("="*70)
