"""
Path utilities for cross-platform support
Handles Windows, Mac, and Linux paths correctly
"""

from pathlib import Path
import os

# Determine the outputs directory based on platform
def get_output_dir():
    """
    Get the output directory path (cross-platform)
    
    Returns:
        Path: Path object for the outputs directory
    """
    # Check if we're in the Linux environment (has /mnt/user-data/outputs)
    linux_path = Path('/mnt/user-data/outputs')
    if linux_path.exists():
        return linux_path
    
    # Otherwise, use local outputs directory
    local_path = Path('./outputs')
    local_path.mkdir(exist_ok=True)
    return local_path


def get_data_dir():
    """
    Get the data directory path (cross-platform)
    
    Returns:
        Path: Path object for the data directory
    """
    # Check for Linux environment
    linux_path = Path('/mnt/user-data/uploads')
    if linux_path.exists():
        return linux_path
    
    # Otherwise, use local data directory
    local_path = Path('./data')
    local_path.mkdir(exist_ok=True)
    return local_path


def ensure_dir(path):
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        path: str or Path object
    
    Returns:
        Path: Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_save_path(filename, subdir=None):
    """
    Get a cross-platform save path for a file
    
    Args:
        filename: Name of the file
        subdir: Optional subdirectory within outputs
    
    Returns:
        Path: Full path where file should be saved
    """
    output_dir = get_output_dir()
    
    if subdir:
        output_dir = output_dir / subdir
        output_dir.mkdir(exist_ok=True)
    
    return output_dir / filename


# Global output directory
OUTPUT_DIR = get_output_dir()
DATA_DIR = get_data_dir()


if __name__ == "__main__":
    print("Path Utilities")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Data directory: {DATA_DIR.absolute()}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Test saving
    test_path = get_save_path('test.txt')
    print(f"Test save path: {test_path}")
