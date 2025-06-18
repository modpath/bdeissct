import os
import shutil


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



source_directory = "/home/azhukova/Demi/anna/projects/bdext/sim_bdeiss/test/500_1000/"
destination_directory = "/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/500_1000/"
filename_to_copy = ".est_bd"

copy_files_with_name(source_directory, destination_directory, filename_to_copy)