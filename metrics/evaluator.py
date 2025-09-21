import pandas as pd

# Đọc file CSV
df = pd.read_csv("summary_0609_0611.csv")

# Nhóm theo 4 cột và tính trung bình
grouped = (
    df.groupby(["num_of_job", "num_of_machine", "splitmin", "algo"])
      .agg(
          avg_percentage_gap=("percentage_gap", "mean"),
          avg_processing_time_ms=("processing_time_ms", "mean"),
          count=("percentage_gap", "count")  # số hàng trong nhóm
      )
      .reset_index()
)

# Xuất ra file mới
grouped.to_csv("summary_results.csv", index=False)

print(grouped.head())
