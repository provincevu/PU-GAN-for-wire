# import numpy as np
# import open3d as o3d
# import os
# from sklearn.cluster import DBSCAN

# # 1. Đọc dữ liệu
# pcd = o3d.io.read_point_cloud("powerline_points_2.xyz")
# points = np.asarray(pcd.points)

# # 2. Phân cụm DBSCAN trên (x, y, z)
# db = DBSCAN(eps=0.3, min_samples=3).fit(points)
# labels = db.labels_
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(f"Số cluster (x, y, z): {n_clusters}")

# MIN_POINTS = 100
# os.makedirs("data_clust", exist_ok=True)
# saved_count = 0
# patch_idx = 1

# for i in range(n_clusters):
#     cluster_points = points[labels == i]
#     if len(cluster_points) < MIN_POINTS:
#         continue   # <-- Chỉ giữ lại cluster đủ lớn
#     np.savetxt(f"data_clust/patch_{patch_idx}.xyz", cluster_points, fmt='%.6f')
#     print(f"Saved patch_{patch_idx}.xyz with {len(cluster_points)} points")
#     patch_idx += 1
#     saved_count += 1

# print(f"Total patches saved: {saved_count}")


import numpy as np
import open3d as o3d
import os
from sklearn.cluster import DBSCAN

def cluster_xyz_points(xyz_file, out_folder, min_points=100, eps=0.3, min_samples=3):
    # 1. Đọc dữ liệu
    pcd = o3d.io.read_point_cloud(xyz_file)
    points = np.asarray(pcd.points)

    # 2. Phân cụm DBSCAN trên (x, y, z)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Số cluster (x, y, z): {n_clusters}")

    os.makedirs(out_folder, exist_ok=True)
    saved_count = 0
    patch_idx = 1

    for i in range(n_clusters):
        cluster_points = points[labels == i]
        if len(cluster_points) < min_points:
            continue   # <-- Chỉ giữ lại cluster đủ lớn
        np.savetxt(f"{out_folder}/patch_{patch_idx}.xyz", cluster_points, fmt='%.6f')
        patch_idx += 1
        saved_count += 1

    print(f"Total patches saved: {saved_count}")