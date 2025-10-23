from dataclasses import dataclass

@dataclass
class WinCheck:
    m: int
    widx: int
    raw: tuple        # (s, e)
    ready: float
    aligned_start: float
    residual: float
    feasible: bool
    reasons: list     # ["residual<split_min", "job_bound_to_other_machine", ...]
    ect: float | None # earliest completion time if place here (dry-run)

def check_window(j, m, widx, win, ready_time, split_min,
                 job_bound_machine, estimate_ect_fn):
    s, e = win
    reasons = []
    aligned = max(s, ready_time)
    residual = e - aligned
    feasible = True

    if job_bound_machine is not None and job_bound_machine != m:
        feasible = False; reasons.append("job_bound_to_other_machine")
    if residual <= 0:
        feasible = False; reasons.append("window_ends_before_ready")
    elif residual < split_min:
        feasible = False; reasons.append("residual<split_min")

    ect = None
    if feasible:
        try:
            # Hàm ước tính thời điểm hoàn tất nếu đặt mảnh tại (aligned, e)
            ect = estimate_ect_fn(j, m, (aligned, e))  # DẠNG DRY-RUN, không mutate state
        except Exception as ex:
            feasible = False; reasons.append(f"ect_error:{ex}")

    return WinCheck(m, widx, (s, e), ready_time, aligned, residual, feasible, reasons, ect)
