import os

def rename_files_in_directory(root_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        # Ensure the path is a directory
        if not os.path.isdir(folder_path):
            continue
        
        for file_name in os.listdir(folder_path):
            # Check if it's a JPEG file
            if file_name.endswith('.jpeg'):
                # Extract the number from the file name
                parts = file_name.split('_')
                number_part = None
                
                # Check for 'flipped' or directly the numeric part
                if parts[-1].startswith('flipped'):
                    number_part = parts[-1].replace('flipped_', '').replace('.jpeg', '')
                else:
                    # Assume the last part contains the number without 'flipped'
                    number_part = parts[-1].replace('.jpeg', '')
                
                # Format the number to two digits
                if number_part and number_part.isdigit():
                    new_name = f"{int(number_part):02d}.jpeg"
                    old_path = os.path.join(folder_path, file_name)
                    new_path = os.path.join(folder_path, new_name)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {file_name} -> {new_name}")

# Specify your root directory path
root_data_directory = '/mnt/storage1/esad/data/HILAL_OCT/organized/'
rename_files_in_directory(root_data_directory)
