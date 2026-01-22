import os
import shutil
from pathlib import Path

def split_data(data_dir='data', train_dir='train', test_dir='test', train_count=10):
    """
    Split data from data folder into train and test folders.

    Args:
        data_dir: Source directory containing numbered subfolders
        train_dir: Destination directory for training data
        test_dir: Destination directory for test data
        train_count: Number of images per folder to put in training set
    """
    data_path = Path(data_dir)
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    # Create train and test directories if they don't exist
    train_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)

    # Get all subdirectories in data folder
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda x: int(x.name))  # Sort numerically

    print(f"Found {len(subdirs)} subdirectories: {[d.name for d in subdirs]}")

    for subdir in subdirs:
        folder_name = subdir.name
        print(f"\nProcessing folder: {folder_name}")

        # Create corresponding subdirectories in train and test
        train_subdir = train_path / folder_name
        test_subdir = test_path / folder_name
        train_subdir.mkdir(exist_ok=True)
        test_subdir.mkdir(exist_ok=True)

        # Get all files in the subfolder and sort them
        files = sorted([f for f in subdir.iterdir() if f.is_file()])
        total_files = len(files)

        print(f"  Total files: {total_files}")
        print(f"  Train: {min(train_count, total_files)} files")
        print(f"  Test: {max(0, total_files - train_count)} files")

        # Split files into train and test
        train_files = files[:train_count]
        test_files = files[train_count:]

        # Copy files to train folder
        for file in train_files:
            dest = train_subdir / file.name
            shutil.copy2(file, dest)

        # Copy files to test folder
        for file in test_files:
            dest = test_subdir / file.name
            shutil.copy2(file, dest)

    print("\n" + "="*50)
    print("Data split completed!")
    print(f"Train data: {train_dir}/")
    print(f"Test data: {test_dir}/")

if __name__ == "__main__":
    split_data()
