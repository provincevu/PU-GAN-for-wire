import numpy as np
import os
import h5py
import shutil


def make_clean_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def xyz_folders_to_h5(
    folder_gt: str = "data/ground_truth",
    folder_input: str = "data/input",
    output_h5_name: str = "train_512_2048.h5",
    dataset_name_gt: str = "poisson_2048",
    dataset_name_input: str = "poisson_512"
):
    """
    Đọc dữ liệu từ hai thư mục chứa các file .xyz (ground_truth và input),
    và lưu vào một file HDF5 với hai dataset.

    Args:
        folder_gt: Thư mục chứa các file ground_truth (.xyz)
        folder_input: Thư mục chứa các file input (.xyz)
        output_h5: Đường dẫn file HDF5 để lưu dữ liệu
        dataset_name_gt: Tên dataset ground_truth trong file HDF5
        dataset_name_input: Tên dataset input trong file HDF5
    """
    output_h5 = "../data/train/" + output_h5_name
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    # ==== Đọc danh sách file .xyz ====
    xyz_files_gt = sorted([f for f in os.listdir(folder_gt) if f.endswith('.xyz')])
    xyz_files_input = sorted([f for f in os.listdir(folder_input) if f.endswith('.xyz')])

    n_files_gt = len(xyz_files_gt)
    n_files_input = len(xyz_files_input)
    assert n_files_gt > 0, "Không tìm thấy file .xyz nào trong thư mục ground_truth!"
    assert n_files_input > 0, "Không tìm thấy file .xyz nào trong thư mục input!"
    assert n_files_gt == n_files_input, "Số lượng file ground_truth và input phải bằng nhau!"

    # ==== Đọc số điểm của 1 file để xác định shape ====
    sample_pc_gt = np.loadtxt(os.path.join(folder_gt, xyz_files_gt[0]))
    sample_pc_input = np.loadtxt(os.path.join(folder_input, xyz_files_input[0]))

    n_points_gt = sample_pc_gt.shape[0]
    n_points_input = sample_pc_input.shape[0]
    assert sample_pc_gt.shape[1] == 3, "File ground_truth xyz phải có 3 cột (x, y, z)"
    assert sample_pc_input.shape[1] == 3, "File input xyz phải có 3 cột (x, y, z)"

    # ==== Đọc toàn bộ dữ liệu về numpy array ====
    data_gt = np.zeros((n_files_gt, n_points_gt, 3), dtype=np.float32)
    data_input = np.zeros((n_files_input, n_points_input, 3), dtype=np.float32)

    for i, (fname_gt, fname_input) in enumerate(zip(xyz_files_gt, xyz_files_input)):
        pc_gt = np.loadtxt(os.path.join(folder_gt, fname_gt))
        pc_input = np.loadtxt(os.path.join(folder_input, fname_input))
        assert pc_gt.shape == (n_points_gt, 3), f"File {fname_gt} không đúng shape!"
        assert pc_input.shape == (n_points_input, 3), f"File {fname_input} không đúng shape!"
        data_gt[i] = pc_gt
        data_input[i] = pc_input

    print(f"Shape dữ liệu ground_truth: {data_gt.shape} (số file, số điểm, 3)")
    print(f"Shape dữ liệu input: {data_input.shape} (số file, số điểm, 3)")

    # ==== Ghi ra file h5 ====
    with h5py.File(output_h5, "a") as f:
        f.create_dataset(dataset_name_gt, data=data_gt)
        f.create_dataset(dataset_name_input, data=data_input)
    print(f"Đã ghi dữ liệu vào {output_h5} với tên dataset '{dataset_name_gt}' và '{dataset_name_input}'.")

if __name__ == "__main__":
    xyz_folders_to_h5()