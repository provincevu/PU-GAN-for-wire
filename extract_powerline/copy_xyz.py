import os
import shutil

def copy_xyz_files(source_folder, destination_folder):
    """
    Copy all .xyz files from source_folder to destination_folder.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(".xyz"):
            src_file = os.path.join(source_folder, filename)
            dst_file = os.path.join(destination_folder, filename)
            shutil.copy2(src_file, dst_file)
    print(f"Copied .xyz files from {source_folder} to {destination_folder}")

if __name__ == "__main__":
    # Thay đổi đường dẫn này cho phù hợp với máy của bạn
    source = r"path/to/source/folder"
    destination = r"path/to/destination/folder"
    copy_xyz_files(source, destination)