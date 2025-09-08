from abc import ABC, abstractmethod
import numpy as np

class AgentBase(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray, mask=None) -> int:
        """Chọn hành động dựa trên trạng thái hiện tại"""
        pass

    @abstractmethod
    def update(self) -> None:
        """Cập nhật thông tin/học sau khi thực hiện hành động"""
        pass
