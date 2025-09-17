import time

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