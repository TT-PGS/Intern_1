from abc import ABC, abstractmethod
from base.model import SchedulingModel
import json

class IOHandler(ABC):
    @abstractmethod
    def get_input(self) -> SchedulingModel:
        pass

    @abstractmethod
    def show_output(self, result: dict):
        pass

class ReadJsonIOHandler(IOHandler):
    def __init__(self, io_json_input_file_path: str):
        super().__init__()
        self.json_path = io_json_input_file_path

    def get_input(self) -> SchedulingModel:
        if not self.json_path:
            raise ValueError("⚠️ 'self.json_path' not found.")

        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise IOError(f"❌ Error reading file JSON: {e}")

        model_data = data.get("model")
        if not model_data:
            raise ValueError("❌ key 'model' not found in file JSON.")

        # Convert key of time_windows from str to int
        time_windows_raw = model_data.get("time_windows", {})
        time_windows = {
            int(k): v for k, v in time_windows_raw.items()
        }

        return SchedulingModel(
            num_jobs=model_data["num_jobs"],
            num_machines=model_data["num_machines"],
            processing_times=model_data["processing_times"],
            split_min=model_data["split_min"],
            time_windows=time_windows
        )
    
    def analyze_schedule(self, job_assignments: dict) -> dict:
        """
        Phân tích lịch gán job từ job_assignments:
        - Trả về tổng makespan
        - Timespan từng job
        - Số lần chia của từng job
        """
        from collections import defaultdict

        job_stats = {}
        max_finish_time = 0

        for job_id, machine_chunks in job_assignments.items():
            chunks = []
            for machine_id, intervals in machine_chunks.items():
                chunks.extend(intervals)
            
            if not chunks:
                continue

            chunks.sort()
            job_start = chunks[0][0]
            job_end = max(end for (_, end) in chunks)
            job_stats[job_id] = {
                "num_chunks": len(chunks),
                "start": job_start,
                "end": job_end,
                "timespan": job_end - job_start,
                "chunks": chunks,
            }

            max_finish_time = max(max_finish_time, job_end)

        return {
            "makespan": max_finish_time,
            "jobs": job_stats
        }

    def show_output(self, result: dict):
        print(f"🎯 Total Reward: {result['total_reward']}")
        print(f"📋 Job Assignments: {result.get('job_assignments', {})}")
        stats = self.analyze_schedule(result.get("job_assignments", {}))

        print(f"\n📊 Schedule Summary:")
        print(f"🕒 Total Makespan: {stats['makespan']}")
        for job_id, info in stats['jobs'].items():
            print(f"  - Job {job_id}: {info['num_chunks']} chunks, timespan {info['timespan']} from {info['start']} to {info['end']}")

