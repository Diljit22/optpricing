import shutil
from pathlib import Path

print("Starting cache cleanup...")
# Find all __pycache__ directories in the current folder and subfolders
for path in Path(".").rglob("__pycache__"):
    if path.is_dir():
        print(f"Deleting: {path}")
        shutil.rmtree(path)

print("Cleanup complete.")