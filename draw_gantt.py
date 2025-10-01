#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw ALL Gantt charts (FCFS, SA, GA, DQN) into ONE image file.

Inputs:
  --config       : dataset JSON (chứa model.time_windows, num_jobs, num_machines, ...)
  --results      : results JSON chứa fcfs_results / sa_results / ga_results
  --dqn          : (tuỳ chọn) result JSON chứa dqn_results
  --out          : đường dẫn file ảnh đầu ra (mặc định: gantt_all.png)

Ví dụ:
python draw_gantt_all.py \
  --config /mnt/data/twspwjp_Jobs_10_Machines_2_Splitmin_3_Index_0.json \
  --results /mnt/data/twspwjp_Jobs_10_Machines_2_Splitmin_3_Index_0.json \
  --dqn /mnt/data/dqn_twspwjp_Jobs_10_Machines_2_Splitmin_3_Index_0.json \
  --out /mnt/data/gantt_all.png
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import math


# -----------------------
# Loading & Normalization
# -----------------------

def load_config(config_fp: str) -> Dict[str, Any]:
    with open(config_fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = data.get("model", {})
    time_windows = {int(k): v for k, v in model.get("time_windows", {}).items()}
    return {
        "time_windows": time_windows,
        "num_jobs": model.get("num_jobs"),
        "num_machines": model.get("num_machines"),
        "split_min": model.get("split_min"),
        "processing_times": model.get("processing_times", []),
    }


def load_results_multi(results_fp: str) -> Dict[str, Any]:
    with open(results_fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "FCFS": data.get("fcfs_results"),
        "SA":   data.get("sa_results"),
        "GA":   data.get("ga_results"),
    }


def load_results_dqn(dqn_fp: Optional[str]) -> Optional[Dict[str, Any]]:
    if not dqn_fp:
        return None
    with open(dqn_fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("dqn_results")

def build_job_colors(algo_dfs: Dict[str, pd.DataFrame], cmap_name: str = "tab20") -> Dict[int, tuple]:
    # gom tất cả job id đang có trong mọi thuật toán
    job_ids = set()
    for df in algo_dfs.values():
        if not df.empty:
            job_ids |= set(df["job"].unique().tolist())
    job_ids = sorted(int(j) for j in job_ids)
    n = max(1, len(job_ids))
    cmap = plt.cm.get_cmap(cmap_name, max(10, n))  # đủ dải màu
    return {j: cmap(i % cmap.N) for i, j in enumerate(job_ids)}

def _text_color_for_rgb(rgba):
    # tính độ sáng tương đối để chọn chữ trắng/đen
    r, g, b = rgba[:3]
    L = 0.2126*r + 0.7152*g + 0.0722*b
    return "white" if L < 0.5 else "black"

def normalize_assignments(assignments: List[Dict[str, Any]], algo: str, segments_key: str) -> pd.DataFrame:
    """
    Chuẩn hoá assignments về DataFrame chung.
    - SA/GA/DQN: segments_key = "segments"
    - FCFS     : segments_key = "plan"
    Mỗi segment dạng [*, start, end] => dùng start, end (2 phần tử cuối).
    """
    rows = []
    for a in assignments:
        job = int(a["job"])
        machine = int(a["machine"])
        for seg_idx, seg in enumerate(a.get(segments_key, [])):
            if len(seg) < 3:
                continue
            start, end = float(seg[-2]), float(seg[-1])
            rows.append({
                "algo": algo,
                "machine": machine,
                "job": job,
                "segment_index": seg_idx,
                "start": start,
                "end": end,
                "duration": end - start,
            })
    df = pd.DataFrame(rows, columns=["algo","machine","job","segment_index","start","end","duration"])
    if not df.empty:
        df = df.sort_values(["machine", "start", "job", "segment_index"]).reset_index(drop=True)
    return df


def build_all_dfs(multi: Dict[str, Any], dqn: Optional[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Trả về map algo -> DataFrame normalized.
    Bỏ qua thuật toán nào không có dữ liệu.
    """
    out = {}
    if multi.get("FCFS"):
        out["FCFS"] = normalize_assignments(multi["FCFS"].get("assignments", []), "FCFS", "plan")
    if multi.get("SA"):
        out["SA"] = normalize_assignments(multi["SA"].get("assignments", []), "SA", "segments")
    if multi.get("GA"):
        out["GA"] = normalize_assignments(multi["GA"].get("assignments", []), "GA", "segments")
    if dqn:
        out["DQN"] = normalize_assignments(dqn.get("assignments", []), "DQN", "segments")
    # loại DF rỗng
    out = {k:v for k,v in out.items() if not v.empty}
    return out


def get_makespan_from_block(block: Optional[Dict[str, Any]]) -> Optional[float]:
    if not block:
        return None
    return block.get("makespan")


# --------------
# Plotting utils
# --------------

def _visible_windows_for_machine(wins, mdf, buffer=2):
    """
    wins: list[(s,e)] của machine m
    mdf : df các đoạn job trên machine m (có cột 'end')
    buffer: vẽ thêm 1-2 windows sau window chứa max_end
    """
    if not wins:
        return []
    if mdf is None or mdf.empty:
        # không có job -> vẽ tối thiểu 1..buffer windows đầu
        k = min(len(wins), max(1, buffer))
        return wins[:k]

    max_end = float(mdf["end"].max())
    cut = 0
    for i, (s, e) in enumerate(wins):
        cut = i
        if e >= max_end:
            break
    cut = min(len(wins), cut + 1 + buffer)  # +1 vì inclusive window chứa max_end
    return wins[:cut]


def _compute_xlim_from_visible(all_visible_wins, df_algo):

    tmins, tmaxs = [], []
    # từ jobs
    if df_algo is not None and not df_algo.empty:
        tmins.append(float(df_algo["start"].min()))
        tmaxs.append(float(df_algo["end"].max()))
    # từ windows đã cắt
    for wins in all_visible_wins:
        for s, e in wins:
            tmins.append(float(s)); tmaxs.append(float(e))
    if not tmins or not tmaxs:
        return (0.0, 1.0)
    lo, hi = min(tmins), max(tmaxs)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return (0.0, 1.0)
    pad = max(1.0, 0.03 * (hi - lo))  # chút đệm bên phải
    return (math.floor(lo - pad), math.ceil(hi + pad))

def plot_one_algo(ax,
                  df_algo: pd.DataFrame,
                  time_windows: Dict[int, List[List[float]]],
                  title: str,
                  job_colors: Dict[int, tuple]) -> None:
    if df_algo is None or df_algo.empty:
        ax.set_title(f"{title} (no data)"); ax.axis("off"); return

    df_algo = df_algo.copy()
    df_algo = df_algo[(df_algo["end"] > df_algo["start"])]

    machines = sorted(df_algo["machine"].unique().tolist())
    lane_h, lane_pad = 6.0, 3.0
    y = 0.0
    yticks, ylabels = [], []

    # --- cắt windows theo dữ liệu job (như bản bạn đang dùng) ---
    def _visible_windows_for_machine(wins, mdf, buffer=2):
        if not wins: return []
        if mdf is None or mdf.empty:
            k = min(len(wins), max(1, buffer)); return wins[:k]
        max_end = float(mdf["end"].max()); cut = 0
        for i, (s, e) in enumerate(wins):
            cut = i
            if e >= max_end: break
        cut = min(len(wins), cut + 1 + buffer)
        return wins[:cut]

    # chuẩn bị xlim
    per_machine_visible = {}
    vis_all = []
    for m in machines:
        mdf = df_algo[df_algo["machine"] == m]
        wins = time_windows.get(int(m), [])
        v = _visible_windows_for_machine(wins, mdf, buffer=2)
        per_machine_visible[m] = v
        vis_all.append(v)

    xlo, xhi = _compute_xlim_from_visible(vis_all, df_algo)
    ax.set_xlim(xlo, xhi)

    # vẽ theo máy
    for m in machines:
        mdf = df_algo[df_algo["machine"] == m]
        y_center = y
        lane_bottom = y_center - lane_h/2.0

        # WINDOWS (nền mờ, full lane)
        for s, e in per_machine_visible[m]:
            ax.add_patch(plt.Rectangle(
                (float(s), lane_bottom), float(e)-float(s), lane_h,
                facecolor="lightblue", alpha=0.08,
                edgecolor="black", linewidth=1.2, zorder=0.5
            ))

        # JOBS (70% lane, màu theo job)
        job_h = lane_h * 0.7
        y_job = y_center - job_h/2.0
        for _, r in mdf.iterrows():
            start, dur = float(r["start"]), float(r["duration"])
            jid = int(r["job"])
            fc = job_colors.get(jid, (1.0, 0.6, 0.0, 1.0))  # fallback orange
            tc = _text_color_for_rgb(fc)
            ax.add_patch(plt.Rectangle(
                (start, y_job), dur, job_h,
                facecolor=fc, alpha=0.95,
                edgecolor="black", linewidth=0.8, zorder=2.0
            ))
            ax.text(start + dur/2.0, y_center, f"J{jid}",
                    ha="center", va="center", fontsize=7,
                    color=tc, zorder=3.0)

        ax.axhline(y_center + (lane_h + lane_pad)/2.0, linewidth=0.5, color="gray", alpha=0.5, zorder=1.0)
        yticks.append(y_center); ylabels.append(f"M{m}")
        y += lane_h + lane_pad

    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Time"); ax.set_title(title, fontsize=11)
    ax.set_ylim(-lane_pad/2, y - lane_pad/2)
    ax.grid(axis="x", linewidth=0.3, alpha=0.3); ax.set_axisbelow(True)


def draw_all_in_one(fig_out: str,
                    algo_dfs: Dict[str, pd.DataFrame],
                    time_windows: Dict[int, List[List[float]]],
                    multi_raw: Dict[str, Any],
                    dqn_raw: Optional[Dict[str, Any]]) -> None:
    order = [a for a in ["FCFS", "SA", "GA", "DQN"] if a in algo_dfs]
    n = len(order)
    if n == 0:
        raise SystemExit("No assignments found in any of the provided result files.")

    # >>> tạo bảng màu job nhất quán giữa mọi subplot
    job_colors = build_job_colors(algo_dfs, cmap_name="tab20")

    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(16, max(6, 5*nrows)),
                             constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    def title_for(algo: str) -> str:
        mk = dqn_raw.get("makespan") if algo == "DQN" and dqn_raw else multi_raw.get(algo, {}).get("makespan")
        return f"{algo} (makespan={mk})" if mk is not None else f"{algo}"

    for idx, algo in enumerate(order):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        plot_one_algo(ax, algo_dfs[algo], time_windows, title_for(algo), job_colors)

    # ẩn axes thừa
    total_axes = nrows * ncols
    for idx in range(n, total_axes):
        r, c = idx // ncols, idx % ncols
        axes[r][c].axis("off")

    # >>> legend (tuỳ chọn): màu cho từng job
    handles = [Patch(facecolor=job_colors[j], edgecolor="black", label=f"J{j}") for j in sorted(job_colors)]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(10, len(handles)), frameon=False)

    os.makedirs(os.path.dirname(fig_out) or ".", exist_ok=True)
    plt.savefig(fig_out, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved combined Gantt: {fig_out}")

# -----------
# Entry point
# -----------

def main():
    qnet = f"qnet_36"
    job = 10
    machine = 4
    splimin = 3
    index = 6
    configs = f"./datasets/90/twspwjp_Jobs_{job}_Machines_{machine}_Splitmin_{splimin}_Index_{index}.json"
    results = f"./metrics/20250609/twspwjp_Jobs_{job}_Machines_{machine}_Splitmin_{splimin}_Index_{index}.json"
    dqn = f"./metrics/90/{qnet}/dqn_twspwjp_Jobs_{job}_Machines_{machine}_Splitmin_{splimin}_Index_{index}.json"
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=configs, help="Path to dataset config JSON (has model.time_windows)")
    ap.add_argument("--results", default=results, help="Path to results JSON with fcfs_results/sa_results/ga_results")
    ap.add_argument("--dqn", default=dqn, help="Path to DQN results JSON (has dqn_results)")
    ap.add_argument("--out", default="schedule_gantt.png", help="Output image path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    multi_raw = load_results_multi(args.results)
    dqn_raw = load_results_dqn(args.dqn)

    algo_dfs = build_all_dfs(multi_raw, dqn_raw)
    draw_all_in_one(args.out, algo_dfs, cfg["time_windows"], multi_raw, dqn_raw)


if __name__ == "__main__":
    main()
