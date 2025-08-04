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

# ✨ Phiên bản mặc định dùng trực tiếp
class SimpleIOHandler(IOHandler):
    def get_input(self) -> SchedulingModel:
        # Cứng input đơn giản, sau có thể đọc từ file JSON/YAML
        return SchedulingModel(
            num_jobs=2,
            num_machines=2,
            processing_times=[5, 5],
            split_min=2,
            time_windows={
                0: [[0, 5], [6, 10]],
                1: [[1, 4], [5, 9]]
            }
        )

    def show_output(self, result: dict):
        print(f"🎯 Total Reward: {result['total_reward']}")
        print(f"📋 Job Assignments: {result.get('job_assignments', {})}")

# Advanced version
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
    
    def show_output(self, result: dict):
        print(f"🎯 Total Reward: {result['total_reward']}")
        print(f"📋 Job Assignments: {result.get('job_assignments', {})}")
