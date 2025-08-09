import pytest

@pytest.fixture
def cfg_two_jobs_two_machines():
    # case “né 5-1”, p=6, split_min=3 => nên chia 3-3
    return {
        "agent": "none",
        "model": {
            "num_jobs": 2,
            "num_machines": 2,
            "processing_times": [6, 5],
            "split_min": 3,
            "time_windows": {
                "0": [[0, 5], [6, 10]],   # tổng 9
                "1": [[1, 4], [5, 9]]     # tổng 7
            }
        }
    }

@pytest.fixture
def cfg_infeasible():
    # tổng sức chứa sau lọc split_min < p => phải fail
    # p=6, split_min=3, chỉ có 1 window dài 5 => không thể
    return {
        "agent": "none",
        "model": {
            "num_jobs": 1,
            "num_machines": 1,
            "processing_times": [6],
            "split_min": 3,
            "time_windows": {
                "0": [[0, 5]]
            }
        }
    }

@pytest.fixture
def cfg_small3():
    # 3 jobs, đủ để GA/SA chạy nhanh
    return {
        "agent": "none",
        "model": {
            "num_jobs": 3,
            "num_machines": 2,
            "processing_times": [6, 4, 5],
            "split_min": 2,
            "time_windows": {
                "0": [[0, 4], [5, 9], [10, 14]],
                "1": [[0, 3], [4, 8], [9, 12]]
            }
        }
    }