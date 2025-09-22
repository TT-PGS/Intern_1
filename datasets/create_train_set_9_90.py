import shutil
import os, glob

# đường dẫn gốc
# file name ex: twspwjp_Jobs_10_Machines_2_Splitmin_3_Index_0
src_folder = "20250609"        # file trong folder A

for i in range(1, 11, 1):
    dst_folder = str(i*9)
    os.makedirs(dst_folder, exist_ok=True)

    for j in range(i):
        file_name_pattern = f"twspwjp_Jobs_10_Machines_*Index_{j}.json"
        for file_path in glob.glob(os.path.join(src_folder, file_name_pattern)):
            shutil.copy(file_path, dst_folder)