import os
import glob

def merge_xyz_files(input_folder, output_file):
    xyz_files = sorted(glob.glob(os.path.join(input_folder, "*.xyz")))
    print(f"Found {len(xyz_files)} .xyz files in {input_folder}")

    with open(output_file, "w") as out_f:
        for file in xyz_files:
            with open(file, "r") as in_f:
                for line in in_f:
                    if line.strip() == "":
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            # Chỉ lấy tọa độ x y z, bỏ qua các cột dư thừa nếu có
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            out_f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
                        except Exception as e:
                            continue
    print(f"Done! Merged all .xyz files into {output_file}")

if __name__ == "__main__":
    input_folder = "data_wire"           # Thay bằng đường dẫn folder chứa các file .xyz của bạn
    output_file = "merged.xyz"           # File đầu ra sau khi gộp
    merge_xyz_files(input_folder, output_file)