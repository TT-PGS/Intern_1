from abc import ABC, abstractmethod
import numpy as np

class AgentBase(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Chọn hành động dựa trên trạng thái hiện tại"""
        pass

    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        """Cập nhật thông tin/học sau khi thực hiện hành động"""
        pass
