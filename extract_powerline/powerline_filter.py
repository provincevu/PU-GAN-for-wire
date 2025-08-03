# import os
# import numpy as np
# from sklearn.decomposition import PCA
# import shutil

# # Tùy chỉnh các thông số dưới đây
# FOLDER_PATH = "./data_clust"      # Đường dẫn đến thư mục chứa các file .xyz
# RADIUS = 0.5                      # Bán kính khối cầu
# OUTPUT_FOLDER = "output"          # Tên thư mục kết quả
# THRESHOLD = 10                    # Ngưỡng density_per_length

# def read_xyz(path):
#     points = []
#     with open(path, 'r') as f:
#         for line in f:
#             vals = line.strip().split()
#             if len(vals) >= 3:
#                 points.append([float(vals[0]), float(vals[1]), float(vals[2])])
#     return np.array(points)

# def pca_length(points):
#     if points.shape[0] < 2:
#         return 0
#     pca = PCA(n_components=1)
#     projected = pca.fit_transform(points)
#     length = projected.max() - projected.min()
#     return length

# def avg_density(points, radius):
#     N = points.shape[0]
#     if N == 0:
#         return 0
#     num_samples = max(1, int(0.5 * N))  # Số điểm ngẫu nhiên = 20% tổng số điểm
#     idxs = np.random.choice(N, num_samples, replace=False)
#     densities = []
#     for idx in idxs:
#         center = points[idx]
#         dists = np.linalg.norm(points - center, axis=1)
#         count = np.sum(dists < radius)
#         densities.append(count)
#     return np.mean(densities)

# def main():
#     # Tạo thư mục output nếu chưa có
#     if not os.path.exists(OUTPUT_FOLDER):
#         os.makedirs(OUTPUT_FOLDER)
#     selected_files = []
#     for fname in os.listdir(FOLDER_PATH):
#         if fname.endswith('.xyz'):
#             path = os.path.join(FOLDER_PATH, fname)
#             pts = read_xyz(path)
#             density = avg_density(pts, RADIUS)
#             length = pca_length(pts)
#             density_per_length = density / length if length > 0 else 0
#             print(f"{fname}: avg_density={density:.2f}, pca_length={length:.2f}, density_per_length={density_per_length:.2f}")
#             # Chỉ lưu file nếu density_per_length < THRESHOLD
#             if density_per_length < THRESHOLD:
#                 shutil.copy(path, os.path.join(OUTPUT_FOLDER, fname))
#                 selected_files.append((fname, density, length, density_per_length))
#     # Ghi thông tin các file được chọn ra file log trong output folder
#     with open(os.path.join(OUTPUT_FOLDER, "filtered_list.txt"), 'w') as f:
#         for fname, density, length, density_per_length in selected_files:
#             f.write(f"{fname}\t{density:.2f}\t{length:.2f}\t{density_per_length:.2f}\n")

# if __name__ == "__main__":
#     main()


import os
import numpy as np
from sklearn.decomposition import PCA
import shutil

def filter_patches_by_density(input_folder, output_folder, radius=0.5, threshold=10):
    def read_xyz(path):
        points = []
        with open(path, 'r') as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) >= 3:
                    points.append([float(vals[0]), float(vals[1]), float(vals[2])])
        return np.array(points)

    def pca_length(points):
        if points.shape[0] < 2:
            return 0
        pca = PCA(n_components=1)
        projected = pca.fit_transform(points)
        length = projected.max() - projected.min()
        return length

    def avg_density(points, radius):
        N = points.shape[0]
        if N == 0:
            return 0
        num_samples = max(1, int(0.5 * N))  # Số điểm ngẫu nhiên = 50% tổng số điểm
        idxs = np.random.choice(N, num_samples, replace=False)
        densities = []
        for idx in idxs:
            center = points[idx]
            dists = np.linalg.norm(points - center, axis=1)
            count = np.sum(dists < radius)
            densities.append(count)
        return np.mean(densities)

    # Tạo thư mục output nếu chưa có
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    selected_files = []
    for fname in os.listdir(input_folder):
        if fname.endswith('.xyz'):
            path = os.path.join(input_folder, fname)
            pts = read_xyz(path)
            density = avg_density(pts, radius)
            length = pca_length(pts)
            density_per_length = density / length if length > 0 else 0
            # Chỉ lưu file nếu density_per_length < threshold
            if density_per_length < threshold:
                shutil.copy(path, os.path.join(output_folder, fname))
                selected_files.append((fname, density, length, density_per_length))
    # Ghi thông tin các file được chọn ra file log trong output folder
    with open(os.path.join(output_folder, "filtered_list.txt"), 'w') as f:
        for fname, density, length, density_per_length in selected_files:
            f.write(f"{fname}\t{density:.2f}\t{length:.2f}\t{density_per_length:.2f}\n")