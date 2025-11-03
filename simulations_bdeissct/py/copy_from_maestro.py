import os
import shutil
from glob import iglob


def copy_files_with_path(source_dir, dest_dir, target_filename):
    """
    Recursively copy only files named target_filename from source_dir
    to dest_dir, preserving the directory structure.
    """

    # Walk through the directory tree
    for file in iglob(os.path.join(source_dir, target_filename)):
        folder = os.path.dirname(file).replace(source_dir, dest_dir)

        # Create the directory in the destination if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        shutil.copy2(file, file.replace(source_dir, dest_dir))
        print(f"Copied: {file} -> {file.replace(source_dir, dest_dir)}")

def copy_files_with_name(source_dir, dest_dir, target_filename):
    """
    Recursively copy only files named target_filename from source_dir
    to dest_dir, preserving the directory structure.
    """

    # Walk through the directory tree
    for root, dirs, files in os.walk(source_dir):
        # Compute the relative path from the source's perspective
        rel_path = os.path.relpath(root, source_dir)

        # Construct the corresponding output directory path
        new_root = os.path.join(dest_dir, rel_path)

        # Create the directory in the destination if it doesn't exist
        os.makedirs(new_root, exist_ok=True)

        # Check each file in the current directory
        for file_name in files:
            if target_filename in file_name:
                # Define full source and destination file paths
                src_file = os.path.join(root, file_name)
                dst_file = os.path.join(new_root, file_name)

                # Copy the file, preserving metadata
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")



source_directory = "/home/azhukova/mPath/anna/projects/bdext/sim_bdeiss/train/2000_5000/"
destination_directory = "/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train/2000_5000/"
filename_to_copy = "*/0/trees.0.*"

copy_files_with_path(source_directory, destination_directory, filename_to_copy)
# copy_files_with_name(source_directory, destination_directory, filename_to_copy)