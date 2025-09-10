import os
import re
import shutil

def rename_directories_numerically(root_dir):
    """
    Rename directories in the specified root directory to numerical values.
    
    Parameters:
    - root_dir (str): The path to the root directory containing folders to rename.
    """ 
    # List all directories in the root directory
    folders = os.listdir(root_dir)
    
    # Sort the folders to ensure consistent ordering
    folders.sort()
    
    # Rename each folder numerically
    for i, folder_name in enumerate(folders):
        folder_path = os.path.join(root_dir, folder_name)
        # Ensure the path is a directory
        if not os.path.isdir(folder_path):
            continue
        
        new_folder_name = f"{i+1622}"
        new_folder_path = os.path.join(root_dir, new_folder_name)
        
        # Rename the folder
        os.rename(folder_path, new_folder_path)
        print(f"Renamed: {folder_name} -> {new_folder_name}")

def rename_files_in_directory(root_dir):
    for folder_name in os.listdir(root_dir):
        try:
            if int(folder_name) < 495:
                continue
        except ValueError:
            print(folder_name)
            continue

        folder_path = os.path.join(root_dir, folder_name)
        # Ensure the path is a directory
        if not os.path.isdir(folder_path):
            continue
        
        for file_name in os.listdir(folder_path):
            # Check if it's a JPEG file
            if file_name.endswith('.png'):
                # Extract the number from the file name
                parts = file_name.split('_')
                print(parts)
                if 'ayna' in file_name:
                    number_part = parts[-2]
                else:
                    number_part = parts[-1].replace('.png', '')
                
                # Check for 'flipped' or directly the numeric part
                # if parts[-1].startswith('flipped'):
                #     number_part = parts[-1].replace('flipped_', '').replace('.png', '')
                # else:
                #     # Assume the last part contains the number without 'flipped'
                #     # number_part = parts[-1].replace('.png', '')
                #     numper_part = parts[-2]
                #     print(number_part)
                
                # Format the number to two digits
                if number_part and number_part.isdigit():
                    new_name = f"{int(number_part):02d}.png"
                    old_path = os.path.join(folder_path, file_name)
                    new_path = os.path.join(folder_path, new_name)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {file_name} -> {new_name}")

def flatten_dataset_structure(src_root='train', dst_root='train_all'):
    os.makedirs(dst_root, exist_ok=True)

    for folder_name in sorted(os.listdir(src_root)):
        folder_path = os.path.join(src_root, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_file = os.path.join(folder_path, filename)
                dst_filename = f"{folder_name}_{filename}"
                dst_file = os.path.join(dst_root, dst_filename)
                shutil.copyfile(src_file, dst_file)

    print(f"Images successfully moved to {dst_root}/")

# Specify your root directory path
root_data_directory = '/home/esad-ugur/Data/OCT/train'
# rename_files_in_directory(root_data_directory)
# rename_directories_numerically(root_data_directory) 
flatten_dataset_structure('/home/esad-ugur/Data/LungCT/test', '/home/esad-ugur/Data/LungCT/test_all')