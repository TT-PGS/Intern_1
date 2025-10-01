import os, glob, json
import csv

def save_rows_to_csv(rows, filename="dqn_stats_test_90.csv"):
    """
    Ghi nhiều dòng dữ liệu ra file CSV

    Args:
        rows (list of tuple): danh sách các bộ giá trị.
        filename (str): tên file CSV cần ghi.
    """
    fields = ["n_job", "Training_Set", "Training_Time_(s)", "Test_Set", "%LB"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)

data_collection = []

input_folder_2 = "dqn"
input_file_2 = "training_time_statistic.json"
with open(os.path.join("metrics", "dqn", input_file_2), "r", encoding="utf-8") as f:
    data_stat = json.load(f)

for i in range(1, 11, 1):
    input_folder_1 = f"qnet_{i*9}"
    key_pattern = f"{input_folder_1}.pt"
    sum_percentage_gap = 0
    arr = []
    for file in glob.glob(os.path.join("metrics", "90", input_folder_1, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        sum_percentage_gap += data["dqn_results"]["percentage_gap"]
        haha = data["dqn_results"]["percentage_gap"]
        arr.append(haha)
        # print(haha)
        # if haha < 50:
        #     print(file)
    # print(f"\n\nfiles in {input_folder_1} has {arr}")
    # print(arr)
    print(f"\nfiles in {input_folder_1} has min_LB: {min(arr)}, max_LB: {max(arr)}\n\n")
    data_collection.append((10, i*9, data_stat[key_pattern]["trainning_time"], 90, sum_percentage_gap/90))

save_rows_to_csv(data_collection)

