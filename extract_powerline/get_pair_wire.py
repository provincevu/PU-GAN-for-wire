import numpy as np
import glob
import os
import csv

def read_xyz_file(filepath):
    points = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except:
                    continue
    return np.array(points)

def get_endpoints(points, k=10):
    # Lấy trung bình k điểm đầu/cuối làm đại diện đầu mút, để ổn định
    if points.shape[0] < k:
        k = points.shape[0] // 2 or 1
    start = np.mean(points[:k], axis=0)
    end = np.mean(points[-k:], axis=0)
    dir_vec = end - start
    return start, end, dir_vec

def point_line_distance(point, line_start, line_end):
    # Tính khoảng cách vuông góc từ point tới đoạn thẳng line_start-line_end
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point - line_start)
    line_unitvec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unitvec)
    # Giới hạn proj_length nằm trong đoạn [0, line_len]
    proj_length = np.clip(proj_length, 0, line_len)
    proj_point = line_start + proj_length * line_unitvec
    return np.linalg.norm(point - proj_point)

def perpendicular_offset(end_i, start_j, dir_j):
    """
    end_i: endpoint của đoạn i (numpy array shape (3,))
    start_j: start point của đoạn j
    dir_j: vector hướng của đoạn j (end_j - start_j)
    Trả về độ lệch vuông góc của end_i với trục đoạn j.
    """
    dir_j_norm = dir_j / (np.linalg.norm(dir_j) + 1e-12)
    vec = end_i - start_j
    proj_len = np.dot(vec, dir_j_norm)
    proj_point = start_j + proj_len * dir_j_norm
    perp_vec = end_i - proj_point
    return np.linalg.norm(perp_vec)

def compute_length(points):
    # Tổng chiều dài đoạn thẳng qua toàn bộ các điểm
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.sum(seg_lens)

def match_segments(segments, files, m=0.2, d=2.0, perp_thresh=0.3, offset_thresh=0.5, lengths=None):
    endpoints = []
    for pts in segments:
        start, end, dir_vec = get_endpoints(pts, k=3)
        endpoints.append({'start': start, 'end': end, 'dir': dir_vec})

    matches = []
    n = len(segments)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Khoảng cách đầu mút
            dist = np.linalg.norm(endpoints[i]['end'] - endpoints[j]['start'])
            # Khoảng cách vuông góc từ end của i đến đoạn j
            perp_dist = point_line_distance(endpoints[i]['end'], endpoints[j]['start'], endpoints[j]['end'])
            # Độ lệch vuông góc với trục của đoạn j
            offset = perpendicular_offset(endpoints[i]['end'], endpoints[j]['start'], endpoints[j]['end'] - endpoints[j]['start'])
            if dist >= m and dist <= d and perp_dist <= perp_thresh and offset <= offset_thresh:
                matches.append((
                    files[i], lengths[i],
                    files[j], lengths[j],
                    dist, perp_dist, offset
                ))
    return matches

def main(input_folder, output_csv, m=0.2, d=2.0, perp_thresh=0.3, offset_thresh=0.5):
    xyz_files = sorted(glob.glob(os.path.join(input_folder, "*.xyz")))
    print(f"Found {len(xyz_files)} .xyz files in {input_folder}")
    segments = [read_xyz_file(f) for f in xyz_files]
    filenames = [os.path.basename(f) for f in xyz_files]
    lengths = [compute_length(pts) for pts in segments]
    matches = match_segments(segments, filenames, m=m, d=d, perp_thresh=perp_thresh, offset_thresh=offset_thresh, lengths=lengths)
    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["segment1", "length_1", "segment2", "length_2", "distance", "perp_distance", "perp_offset"])
        for seg1, len1, seg2, len2, dist, perp, offset in matches:
            writer.writerow([seg1, f"{len1:.4f}", seg2, f"{len2:.4f}", f"{dist:.4f}", f"{perp:.4f}", f"{offset:.4f}"])
    print(f"get_pair_wire, Done!")

if __name__ == "__main__":
    # Ví dụ sử dụng
    input_folder = "../data/test"
    output_csv = "../data/test/matched_segments.csv"
    m = 0.0           # Khoảng cách tối thiểu để coi là bị đứt (đơn vị cùng với xyz)
    d = 20.0          # Khoảng cách tối đa để ghép nối
    perp_thresh = 20  # Ngưỡng khoảng cách vuông góc tới đoạn (tùy dữ liệu)
    offset_thresh = 0.5 # Ngưỡng độ lệch vuông góc với trục dây (tùy dữ liệu, nên < khoảng cách giữa 2 dây song song)

    main(input_folder, output_csv, m, d, perp_thresh, offset_thresh)