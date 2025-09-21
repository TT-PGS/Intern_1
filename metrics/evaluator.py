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

# ================== tạo TABLE III ==================

# 1) Đọc CSV
df = pd.read_csv("summary_results.csv")

# 2) Đổi tên metric cho gọn & đúng nhãn bảng
metric_map = {
    "avg_percentage_gap": "%LB",
    "avg_processing_time_ms": "t_ms",
}
# Chỉ giữ cột cần
df = df[["num_of_job","num_of_machine","splitmin","algo",
         "avg_percentage_gap","avg_processing_time_ms"]]

# 3) Chuyển từ wide → long theo 2 metric, rồi pivot lại thành bảng nhiều cột theo algo × metric
long_df = df.melt(
    id_vars=["num_of_job","num_of_machine","splitmin","algo"],
    value_vars=["avg_percentage_gap","avg_processing_time_ms"],
    var_name="metric", value_name="value"
)
long_df["metric"] = long_df["metric"].map(metric_map)

# Đảm bảo thứ tự cột con: %LB trước, rồi t_ms
long_df["metric"] = pd.Categorical(long_df["metric"], categories=["%LB","t_ms"], ordered=True)

# 4) Pivot: index = (n, k, spm); columns = (algo, metric)
pivot = (long_df
         .pivot_table(index=["num_of_job","num_of_machine","splitmin"],
                      columns=["algo","metric"], values="value", aggfunc="first")
         .sort_index()
)

# 5) Sắp xếp thuật toán theo thứ tự FCFS, SA, GA, TODO: DQN
algo_order = [a for a in ["FCFS","SA","GA","DQN"] if a in pivot.columns.get_level_values(0)]
pivot = pivot.reindex(columns=pd.MultiIndex.from_product([algo_order, ["%LB","t_ms"]]), fill_value=None)

# 6) Làm gọn tên cột (FCFS_%LB, FCFS_t_ms, SA_%LB, SA_t_ms, ...)
pivot.columns = [f"{algo}_{metric}" for algo, metric in pivot.columns]

# 7) Làm tròn cho đẹp (ví dụ %LB giữ 2 chữ số sau dấu phẩy, t_ms giữ 2)
round_map = {c: 2 for c in pivot.columns if c.endswith("%LB")}
round_map.update({c: 2 for c in pivot.columns if c.endswith("_t_ms")})
pivot = pivot.round(round_map)

# 8) Xuất ra CSV / LaTeX
pivot.to_csv("table_like_paper.csv")               # dạng csv phẳng
latex = pivot.to_latex(index=True)                 # LaTeX đơn giản
with open("table_like_paper.tex","w",encoding="utf-8") as f:
    f.write(latex)

print(pivot.head())

