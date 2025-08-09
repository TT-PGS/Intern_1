from staticAlgos.SA import solve_with_SA
from staticAlgos.FCFS import schedule_fcfs

def test_sa_returns_better_or_equal_than_fcfs(cfg_small3):
    # baseline FCFS
    fcfs_sched, fcfs_ms = schedule_fcfs(cfg_small3, job_order=None)
    # SA (tham số bài báo + iters_per_temp = số job mặc định)
    result = solve_with_SA(
        cfg_small3,
        Tmax=500.0,
        Tthreshold=1.0,
        alpha=0.99,
        random_seed=42
    )
    assert "best_makespan" in result and result["best_makespan"] is not None
    # SA có thể bằng hoặc tốt hơn FCFS (vì bắt đầu random + nhận nghiệm kém với xác suất)
    assert result["best_makespan"] <= fcfs_ms