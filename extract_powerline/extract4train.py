import os
import sys
import random
import numpy as np
from sklearn.decomposition import PCA
import shutil
import re
from xyz2h5 import make_clean_folder
from extract_from_las import extract_wire_from_las
from xyz2h5 import xyz_folders_to_h5

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Common.pc_util import normalize_point_cloud

def cut_random_center(points, cut_ratio=0.75, head_tail_ratio=0.02):
    N = points.shape[0]
    cut_len = int(N * cut_ratio)
    head_tail_min = int(N * head_tail_ratio)
    if cut_len < 1 or N < 2 * head_tail_min + cut_len:
        return None
    min_start = head_tail_min
    max_start = N - cut_len - head_tail_min
    if max_start < min_start:
        return None
    cut_start = random.randint(min_start, max_start)
    keep_idx = np.concatenate([np.arange(0, cut_start), np.arange(cut_start + cut_len, N)])
    return points[keep_idx]

def get_next_idx(folder, prefix):
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.xyz')]
    if not files:
        return 0
    numbers = []
    for f in files:
        match = re.search(rf"{prefix}_(\d+)\.xyz", f)
        if match:
            numbers.append(int(match.group(1)))
    if not numbers:
        return 0
    return max(numbers) + 1

def process_patches(input_folder, output_input_folder, output_gt_folder, patch_gt_size=2048, stride=None, cut_ratio=0.75, head_tail_ratio=0.05, start_patch_idx=0):
    if stride is None:
        stride = patch_gt_size // 4
    random.seed(42)
    os.makedirs(output_input_folder, exist_ok=True)
    os.makedirs(output_gt_folder, exist_ok=True)

    patch_idx = start_patch_idx
    patch_count = 0

    for file in sorted(os.listdir(input_folder)):
        if not file.endswith('.xyz'):
            continue
        patch_points = np.loadtxt(os.path.join(input_folder, file))
        n = len(patch_points)
        for start in range(0, n - patch_gt_size + 1, stride):
            pts = patch_points[start:start + patch_gt_size]
            if pts.shape[0] < patch_gt_size:
                continue
            pca = PCA(n_components=1)
            proj = pca.fit_transform(pts).flatten()
            sort_idx = np.argsort(proj)
            proj_sorted = proj[sort_idx]
            if proj_sorted[0] > proj_sorted[-1]:
                sort_idx = sort_idx[::-1]
            pts_sorted = pts[sort_idx]

            patch_input = cut_random_center(pts_sorted, cut_ratio=cut_ratio, head_tail_ratio=head_tail_ratio)
            if patch_input is None:
                continue

            patch_gt_norm, centroid, scale = normalize_point_cloud(pts_sorted)
            patch_input_norm = (patch_input - centroid) / scale
            np.savetxt(f'{output_input_folder}/patch_input_{patch_idx:06d}.xyz', patch_input_norm, fmt='%.8f')
            np.savetxt(f'{output_gt_folder}/patch_gt_{patch_idx:06d}.xyz', patch_gt_norm, fmt='%.8f')
            patch_idx += 1
            patch_count += 1

    print(f"Đã lưu {patch_count} patch input và ground truth từ folder {input_folder}.")
    return patch_idx

if __name__ == "__main__":
    las_folder = "../data"
    patch_gt_size = 2048 # Kích thước patch ground truth
    z_min = 30 # Ngưỡng độ cao z tối thiểu để giữ điểm
    cut_ratio = 0.75 # Tỷ lệ cắt patch, cắt đi 1 - intput_size/patch_gt_size

    # Folder cho kết quả cuối cùng
    output_input_folder = '../data/input'
    output_gt_folder = '../data/ground_truth'

    # Chỉ xóa folder output 1 lần duy nhất trước khi bắt đầu xử lý tất cả LAS
    make_clean_folder(output_input_folder)
    make_clean_folder(output_gt_folder)

    patch_idx = 0

    for las_file in sorted(os.listdir(las_folder)):
        if not las_file.lower().endswith('.las'):
            continue
        abs_las_file = os.path.join(las_folder, las_file)
        print(f"Đang xử lý {abs_las_file}")

        patch_folder = os.path.join(las_folder, f"patches_{os.path.splitext(las_file)[0]}")
        filtered_patch_folder = os.path.join(las_folder, f"filtered_patches_{os.path.splitext(las_file)[0]}")

        make_clean_folder(patch_folder)
        make_clean_folder(filtered_patch_folder)

        extract_wire_from_las(abs_las_file, patch_folder, filtered_patch_folder, z_min=z_min)

        prev_patch_idx = patch_idx
        patch_idx = process_patches(
            filtered_patch_folder,
            output_input_folder,
            output_gt_folder,
            patch_gt_size=patch_gt_size,
            cut_ratio=cut_ratio, # Tỷ lệ cắt patch, cắt đi 1 - intput_size/patch_gt_size
            start_patch_idx=patch_idx
        )
        print(f"Đã xử lý {patch_idx - prev_patch_idx} patch từ file {las_file}.")

    # Sau khi hoàn tất tất cả file .las, gom về h5
    ratio = (1-cut_ratio)**-1
    output_h5_name = f"train_{int(patch_gt_size//ratio)}_{patch_gt_size}.h5"

    xyz_folders_to_h5(
        folder_gt=output_gt_folder,
        folder_input=output_input_folder,
        output_h5_name=output_h5_name,
        dataset_name_gt=f"poisson_{patch_gt_size}",
        dataset_name_input=f"poisson_{patch_gt_size//4}"
    )

    print(f"Đã lưu file HDF5 vào: {output_h5_name}")