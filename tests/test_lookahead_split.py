from staticAlgos.LookAheadSplit import lookahead_split

def test_split_3_3(cfg_two_jobs_two_machines):
    model = cfg_two_jobs_two_machines["model"]
    # thử trên machine 0 cho job 0 (p=6, split_min=3)
    subs = lookahead_split(
        job_index=0,
        processing_times=model["processing_times"],
        split_min=model["split_min"],
        machine_id=0,
        time_windows=model["time_windows"]
    )
    assert subs, "LookAheadSplit phải trả về phân mảnh hợp lệ"
    total = sum(sj["end"] - sj["start"] for sj in subs)
    assert total == model["processing_times"][0], "Tổng thời lượng subjobs phải bằng p"
    # tránh “5-1”: mọi mảnh đều >= split_min
    assert min(sj["end"] - sj["start"] for sj in subs) >= model["split_min"]

def test_respect_split_min(cfg_two_jobs_two_machines):
    model = cfg_two_jobs_two_machines["model"]
    subs = lookahead_split(
        job_index=1,  # p=5, split_min=3
        processing_times=model["processing_times"],
        split_min=model["split_min"],
        machine_id=1,
        time_windows=model["time_windows"]
    )
    # có thể không luôn xếp được trên machine 1 tuỳ cửa sổ;
    # nếu xếp được thì mọi mảnh đều >= split_min
    if subs:
        assert min(sj["end"] - sj["start"] for sj in subs) >= model["split_min"]

def test_infeasible_total_capacity(cfg_infeasible):
    model = cfg_infeasible["model"]
    subs = lookahead_split(
        job_index=0,
        processing_times=model["processing_times"],
        split_min=model["split_min"],
        machine_id=0,
        time_windows=model["time_windows"]
    )
    assert subs == [], "Không đủ tổng sức chứa → phải trả []"