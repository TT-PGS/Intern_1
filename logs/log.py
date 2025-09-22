import time
import os
from datetime import datetime

def log_info(func):
    def wrapper(*args, **kwargs):
        """Decorator để log thông tin hàm gọi."""
        print(f"Calling function: {func.__name__}")
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"Function {func.__name__} completed in {duration:.4f} seconds")
        return result
    return wrapper

def write_log(log: str, log_name = "log.log") -> None:
    log_file = os.path.join(".", log_name)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {log}\n")