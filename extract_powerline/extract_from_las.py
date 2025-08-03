from powerline_extraction import extract_powerline_points
from powerline_clustering import cluster_xyz_points
from powerline_filter import filter_patches_by_density
import os

def extract_wire_from_las(las_file, patch_folder, output_folder, z_min=30):
    # Trích xuất điểm nghi là dây điện từ file .las
    extract_data_folder = "extract_data"

    xyz_powerline = f"{extract_data_folder}/powerline_points_2.xyz"
    extract_powerline_points(las_file, xyz_powerline, z_min)

    # Phân cụm các điểm dây điện thành các patch nhỏ
    os.makedirs(patch_folder, exist_ok=True)
    cluster_xyz_points(xyz_powerline, patch_folder, min_points=100)

    # Lọc các patch theo mật độ/density_per_length
    os.makedirs(output_folder, exist_ok=True)
    filter_patches_by_density(patch_folder, output_folder, radius=0.5, threshold=10)

if __name__ == "__main__":
    las_file = "../data/cloud6ee7a744878ffdc8_Block_3.las"
    patch_folder = "patches"
    output_folder = "filtered_patches"
    extract_wire_from_las(las_file, patch_folder, output_folder, z_min=30)