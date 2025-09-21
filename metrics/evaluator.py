"""
Trích xuất results thành file csv
"""

import csv
import json
import os
from glob import glob
from typing import Any, Dict, Iterable, List, Optional

#========================== helpers ==========================

def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse JSON: {path} ({e})")
        return None

#========================== main process ==========================
i = 0

results_folder = "metrics"

for file_name in glob(os.path.join(results_folder, "**", "*.json"), recursive=True):
    print(file_name)
    i += 1
print(f"total files: {i}")