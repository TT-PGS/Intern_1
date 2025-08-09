from staticAlgos.GA import run_ga
from staticAlgos.FCFS import schedule_fcfs

def test_ga_beats_fcfs_small_cfg(cfg_small3):
    fcfs_sched, fcfs_ms = schedule_fcfs(cfg_small3, job_order=None)
    best_perm, schedule, ms = run_ga(
        cfg_small3,
        pop_size=20,        # nhẹ
        generations=50,     # nhẹ
        tourn_size=5,
        cx_prob=0.8,
        mut_prob=0.1,
        seed=123
    )
    assert best_perm and schedule and ms is not None
    # Kỳ vọng GA không tệ hơn FCFS trong bài toán nhỏ
    assert ms <= fcfs_ms
    # Hoán vị hợp lệ
    assert sorted(best_perm) == list(range(cfg_small3["model"]["num_jobs"]))