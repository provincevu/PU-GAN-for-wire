# import laspy
# import numpy as np
# from sklearn.decomposition import PCA

# # 1. Đọc file .las
# las = laspy.read('cloud6ee7a744878ffdc8_Block_3.las')
# points = np.vstack((las.x, las.y, las.z)).transpose()

# # 2. Chia point cloud thành voxel hình hộp chữ nhật
# def create_rect_voxel_grid(points, voxel_size):
#     vx, vy, vz = voxel_size
#     voxel_indices = np.floor(points / [vx, vy, vz]).astype(int)
#     from collections import defaultdict
#     voxels = defaultdict(list)
#     for idx, v in enumerate(voxel_indices):
#         voxels[tuple(v)].append(idx)
#     return voxels

# voxel_size = (5.0, 5.0, 2.0)  # x, y, z
# voxels = create_rect_voxel_grid(points, voxel_size)

# # 3. Tính PCA cho từng voxel, lấy linearity
# results = []
# for voxel_idx, indices in voxels.items():
#     if len(indices) < 100:  # Loại bỏ voxel quá ít điểm
#         continue
#     pts = points[indices]
#     pca = PCA(n_components=3)
#     pca.fit(pts)
#     lam = pca.singular_values_ ** 2
#     lam.sort()
#     lam = lam[::-1]  # lam1 >= lam2 >= lam3
#     linearity = (lam[0] - lam[1]) / lam[0] if lam[0] > 0 else 0
#     if linearity > 0.85:  # Ngưỡng linearity, có thể điều chỉnh
#         results.append({
#             'voxel_idx': voxel_idx,
#             'indices': indices,
#             'linearity': linearity,
#             'direction': pca.components_[0],  # vector phương dây điện
#         })

# # 4. Lưu điểm thuộc voxel tuyến tính ra file mới, chỉ giữ điểm z >= 30
# powerline_indices = [idx for r in results for idx in r['indices']]
# powerline_points = points[powerline_indices]
# # Lọc theo điều kiện z >= 30
# powerline_points = powerline_points[powerline_points[:, 2] >= 30]

# # === GHI RA FILE XYZ ===
# np.savetxt('powerline_points_2.xyz', powerline_points, fmt="%.6f")
# print(f"Lưu {len(powerline_points)} điểm nghi là dây điện (z >= 30) vào powerline_points_2.xyz")



import laspy
import numpy as np
from sklearn.decomposition import PCA

def extract_powerline_points(las_path, out_xyz_path, z_min=30):
    # 1. Đọc file .las
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # 2. Chia point cloud thành voxel hình hộp chữ nhật
    def create_rect_voxel_grid(points, voxel_size):
        vx, vy, vz = voxel_size
        voxel_indices = np.floor(points / [vx, vy, vz]).astype(int)
        from collections import defaultdict
        voxels = defaultdict(list)
        for idx, v in enumerate(voxel_indices):
            voxels[tuple(v)].append(idx)
        return voxels

    voxel_size = (5.0, 5.0, 2.0)  # x, y, z
    voxels = create_rect_voxel_grid(points, voxel_size)

    # 3. Tính PCA cho từng voxel, lấy linearity
    results = []
    for voxel_idx, indices in voxels.items():
        if len(indices) < 100:  # Loại bỏ voxel quá ít điểm
            continue
        pts = points[indices]
        pca = PCA(n_components=3)
        pca.fit(pts)
        lam = pca.singular_values_ ** 2
        lam.sort()
        lam = lam[::-1]  # lam1 >= lam2 >= lam3
        linearity = (lam[0] - lam[1]) / lam[0] if lam[0] > 0 else 0
        if linearity > 0.85:  # Ngưỡng linearity, có thể điều chỉnh
            results.append({
                'voxel_idx': voxel_idx,
                'indices': indices,
                'linearity': linearity,
                'direction': pca.components_[0],  # vector phương dây điện
            })

    # 4. Lưu điểm thuộc voxel tuyến tính ra file mới, chỉ giữ điểm z >= z_min
    powerline_indices = [idx for r in results for idx in r['indices']]
    powerline_points = points[powerline_indices]
    # Lọc theo điều kiện z >= z_min
    powerline_points = powerline_points[powerline_points[:, 2] >= z_min]

    # === GHI RA FILE XYZ ===
    np.savetxt(out_xyz_path, powerline_points, fmt="%.6f")
    print(f"Lưu {len(powerline_points)} điểm nghi là dây điện (z >= {z_min}) vào {out_xyz_path}")