from collections import defaultdict
from staticAlgos.FCFS import schedule_fcfs

def _job_to_machines(schedule):
    d = defaultdict(set)
    for sj in schedule:
        d[sj["job_id"]].add(sj["machine_id"])
    return d

def test_fcfs_valid_and_jobperson(cfg_two_jobs_two_machines):
    sched, ms = schedule_fcfs(cfg_two_jobs_two_machines, job_order=None)
    assert isinstance(ms, int) or isinstance(ms, float)
    # job-person: mỗi job chỉ xuất hiện ở 1 máy
    jm = _job_to_machines(sched)
    assert all(len(mset) == 1 for mset in jm.values())

def test_fcfs_deterministic(cfg_two_jobs_two_machines):
    s1, m1 = schedule_fcfs(cfg_two_jobs_two_machines, job_order=None)
    s2, m2 = schedule_fcfs(cfg_two_jobs_two_machines, job_order=None)
    # cùng input → phải giống nhau
    assert m1 == m2 and s1 == s2