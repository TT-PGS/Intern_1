from abc import ABC, abstractmethod
from base.model import SchedulingModel

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
