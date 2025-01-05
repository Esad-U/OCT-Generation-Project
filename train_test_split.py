import os
import shutil
import random

def split_folders(root_dir, test_size=20):
    """
    Copies folders from root directory into test and train directories
    while preserving the original structure.
    
    Args:
        root_dir (str): Path to the root directory containing folders to split
        test_size (int): Number of folders to copy to test directory
    """
    # Get immediate subdirectories
    subdirs = [d for d in os.listdir(root_dir) 
              if os.path.isdir(os.path.join(root_dir, d))]
    
    if len(subdirs) < test_size:
        raise ValueError(f"Root directory contains fewer than {test_size} folders")
    
    # Randomly select folders for test set
    test_folders = random.sample(subdirs, test_size)
    train_folders = [d for d in subdirs if d not in test_folders]
    
    # Create test and train directories if they don't exist
    parent_dir = os.path.dirname(root_dir)
    test_dir = os.path.join(parent_dir, 'test')
    train_dir = os.path.join(parent_dir, 'train')
    
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    
    # Copy folders to respective directories
    for folder in test_folders:
        src = os.path.join(root_dir, folder)
        dst = os.path.join(test_dir, folder)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Copied {folder} to test directory")
    
    for folder in train_folders:
        src = os.path.join(root_dir, folder)
        dst = os.path.join(train_dir, folder)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Copied {folder} to train directory")
    
    print(f"\nComplete! Copied {len(test_folders)} folders to test and "
          f"{len(train_folders)} folders to train")

# Example usage
if __name__ == "__main__":
    # Replace with your root directory path
    root_directory = "/mnt/storage1/esad/data/HILAL_OCT/organized"
    split_folders(root_directory)