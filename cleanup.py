"""Clean up temporary and redundant files from the project."""
import os
import shutil
from pathlib import Path

def clean_project():
    """Clean up temporary and redundant files."""
    project_root = Path(__file__).parent
    
    # 1. Remove __pycache__ directories
    for root, dirs, _ in os.walk(project_root):
        if "__pycache__" in dirs:
            cache_dir = os.path.join(root, "__pycache__")
            print(f"Removing: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)
    
    # 2. Remove log files
    log_files = [
        project_root / "bot_debug.log",
        project_root / "logs" / "trading_system.log"
    ]
    
    for log_file in log_files:
        if log_file.exists():
            print(f"Removing: {log_file}")
            os.unlink(log_file)
    
    # 3. Remove any remaining .pyc files
    for pyc_file in project_root.rglob("*.pyc"):
        print(f"Removing: {pyc_file}")
        os.unlink(pyc_file)
    
    # 4. Clean up data directory
    data_dir = project_root / "data"
    if data_dir.exists():
        # Remove all files in shared_buffer
        shared_buffer = data_dir / "shared_buffer"
        if shared_buffer.exists():
            print(f"Cleaning up: {shared_buffer}")
            for item in shared_buffer.glob("*"):
                if item.is_file():
                    item.unlink()
    
    print("\nCleanup completed!")

if __name__ == "__main__":
    clean_project()
