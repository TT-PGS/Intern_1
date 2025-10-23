import time
import os, json
from datetime import datetime

def log_info(func):
    def wrapper(*args, **kwargs):
        """Decorator để log thông tin hàm gọi."""
        print(f"\nCalling function: {func.__name__}")
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"Function {func.__name__} completed in {duration:.4f} seconds")
        return result
    return wrapper

def write_log(log: str, log_name = "log.log") -> None:
    return
    log_file = os.path.join(".", log_name)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {log}\n")

def write_jsonl(obj, log_name="trace_steps.jsonl", add_ts=True, step=None):
    """
    Ghi 1 dòng JSON đúng chuẩn vào file log JSONL.
    - obj: dict hoặc bất kỳ dữ liệu JSON-serializable nào
    - add_ts: nếu True, thêm timestamp vào bên trong JSON
    - step: nếu có, thêm trường step vào JSON
    """
    return
    log_file = os.path.join(".", log_name)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # Nếu obj là str, cố gắng parse sang dict
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            obj = {"msg": obj}

    if not isinstance(obj, dict):
        obj = {"value": obj}

    if add_ts:
        obj["ts"] = datetime.now().isoformat(timespec="seconds")
    if step is not None:
        obj["step"] = int(step)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")