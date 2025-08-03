import os
from extract_from_las import extract_wire_from_las
from get_pair_wire import main as match_segments_main
from xyz2h5 import make_clean_folder
from copy_xyz import copy_xyz_files


def find_las_file(folder):
    las_files = [f for f in os.listdir(folder) if f.lower().endswith('.las')]
    if not las_files:
        raise FileNotFoundError(f"No .las file found in {folder}")
    if len(las_files) > 1:
        raise RuntimeError(f"More than one .las file found in {folder}: {las_files}")
    return os.path.join(folder, las_files[0])

def remove_xyz_files(folder):
    for f in os.listdir(folder):
        if f.lower().endswith('.xyz'):
            try:
                os.remove(os.path.join(folder, f))
            except Exception as e:
                print(f"Could not remove {f}: {e}")

if __name__ == "__main__":
    data_test_folder = "../data/test"
    las_file = find_las_file(data_test_folder)
    remove_xyz_files(data_test_folder)

    patch_folder = "../data/test/temp"
    output_folder = data_test_folder
    z_min = 30
    extract_wire_from_las(las_file, patch_folder, output_folder, z_min=z_min)
    copy_xyz_files(output_folder, "../data/test/output")
    make_clean_folder(patch_folder)

    input_folder = data_test_folder
    output_csv = os.path.join(data_test_folder, "matched_segments.csv")

    m = 0.0           # Khoảng cách tối thiểu để coi là bị đứt
    d = 20.0          # Khoảng cách tối đa để ghép nối

    perp_thresh = 20  # Ngưỡng khoảng cách vuông góc tới đoạn
    offset_thresh = 0.5 # Ngưỡng độ lệch vuông góc với trục dây

    match_segments_main(input_folder, output_csv, m, d, perp_thresh, offset_thresh)

    print(f"Done! Kết quả ghép nối đã lưu ở {output_csv}")