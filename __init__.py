import time

def log_info(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"duration: {duration}")
        return result

    return wrapper