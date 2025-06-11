import os
import shutil


def copy_files_with_name(source_dir, dest_dir):
    """
    Copy DL model files
    """

    for item in os.listdir(source_dir):
        source_item_path = os.path.join(source_dir, item)

        # Check if this item is actually a directory
        if os.path.isdir(source_item_path):
            # Create a matching directory in the destination if needed
            dest_item_path = os.path.join(dest_dir, item)
            os.makedirs(dest_item_path, exist_ok=True)

            # Now copy only text files from that source directory
            for file_name in os.listdir(source_item_path):
                source_file = os.path.join(source_item_path, file_name)

                # Check that this is a file (not another folder) and ends with '.txt'
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, dest_item_path)
                    print(f"Copied: {source_file} to {dest_item_path}")


source_directory = "/home/azhukova/Demi/anna/projects/bdext/sim_bdeiss/training/500_1000/"
# destination_directory = "/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/training/500_1000/"
destination_directory = "/home/azhukova/projects/bdeissct_dl/bdeissct_dl/models"

copy_files_with_name(source_directory, destination_directory)