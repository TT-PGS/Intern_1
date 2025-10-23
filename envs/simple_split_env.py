import gymnasium as gym
import numpy as np

from envs.common_debug import check_window
import json, time, math

from typing import Tuple, Optional
from logs.log import write_log, write_jsonl

from base.model import SchedulingModel
from split_job.dp_single_job import (
    solve_min_timespan_cfg,
    solve_feasible_leftover_rule_cfg,
    assign_job_to_machine,
)

def debug_list_windows(j, windows_by_m, ready_time_j, split_min,
                       job_bound_machine, makespan_now, estimate_ect_fn,
                       file_obj=None):
    seen = []
    for m, wins in windows_by_m.items():
        for widx, w in enumerate(wins):
            seen.append(check_window(
                j, m, widx, w, ready_time_j, split_min,
                job_bound_machine, makespan_now, estimate_ect_fn
            ))
    # Tổng hợp theo máy
    per_m = {}
    for r in seen:
        per_m.setdefault(r.m, []).append(r)

    # Ghi ra JSON 1 dòng để dễ grep theo thời gian
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    out = {
        "ts": stamp,
        "phase": "pre_action_mask",
        "job": int(j),
        "ready": float(ready_time_j),
        "split_min": float(split_min),
        "machines": {}
    }
    for m, arr in per_m.items():
        out["machines"][int(m)] = [{
            "widx": r.widx,
            "raw": list(r.raw),
            "aligned_start": r.aligned_start,
            "residual": r.residual,
            "feasible": r.feasible,
            "reasons": r.reasons,
            "ect": r.ect
        } for r in arr]
    line = json.dumps(out, ensure_ascii=False)
    if file_obj: 
        file_obj.write(line+"\n"); file_obj.flush()
    else:
        print(line)
    return per_m

class SimpleSplitSchedulingEnv(gym.Env):
    """
    Fixed-dim env via padding:
      - Observation: [jobs_remaining_pad (N_MAX), machines_finish_pad (M_MAX)]
      - Action: Discrete(N_MAX * M_MAX), decode to (job, machine) with M_MAX
      - Mask invalid (pad or infeasible) actions to 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        model: SchedulingModel,
        mode: str = "assign",
        verbose: bool = True,
        n_max: int = 50,
        m_max: int = 4,
        w_max: int = 24,
        time_scale: Optional[float] = None,
        max_job_size: Optional[float] = 24,
        alpha: float = 0.15,
        beta: float = 0,
        lam: float = 15,
        radius_ratio: float = 0.0,
        radius_min: float = 0,
        topK: Optional[int] = 5,
    ):
        super().__init__()

        self._blocked_actions = set()
        # --- Real instance data
        self.model = model
        self.real_num_jobs = model.get_num_jobs()
        self.real_num_machines = model.get_num_machines()
        self.processing_times = model.get_processing_times()
        self.time_windows = model.get_time_windows()
        self.split_min = model.get_split_min()
        self.lower_bound = model.get_lower_bound()

        self.radius_ratio = float(radius_ratio)
        self.radius_min = float(self.lower_bound/8)
        if radius_min:
            self.radius_min = float(radius_min)
        self.topK = int(topK) if topK is not None else None

        # Tham số reward shaping
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lam = float(lam)

        # --- Fixed caps (padding targets)
        self.N_MAX = n_max if n_max is not None else self.real_num_jobs
        self.M_MAX = m_max if m_max is not None else self.real_num_machines
        self.W_MAX = w_max
        assert self.N_MAX >= self.real_num_jobs and self.M_MAX >= self.real_num_machines, \
            f"N_MAX {self.N_MAX}/M_MAX {self.M_MAX} must cover the real instance size {self.real_num_jobs} and {self.real_num_machines}."

        self.earliest_win_start = None
        self._tol = 1e-6

        # --- Train/eval controls
        self.verbose = verbose
        self.mode = mode

        # --- Dynamics
        self.current_makespan = 0
        self.done = False
        self.jobs_remaining = []
        self.machines_finish = []
        self.job_assignments = {}

        obs_dim = self.N_MAX + self.M_MAX * (2 * self.W_MAX)
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.N_MAX * self.M_MAX * self.W_MAX)

        # --- step cap (dùng kích thước THỰC để không kéo dài vô hạn)
        self.max_steps = self.real_num_jobs * max(1, self.real_num_machines) * 50
        self.time_scale = float(time_scale) if time_scale else max(1.0, self._max_window_end())
        self.max_job_size = float(max_job_size) if max_job_size is not None else float(max(self.processing_times) or 1.0)

        self._fallback_used = False

    # ----------------- Helpers -----------------

    def build_action_mask(self):
        eps = 1e-9
        mask = np.zeros((self.N_MAX, self.M_MAX, self.W_MAX), dtype=bool)
        pref = np.zeros_like(mask)  # preferred mask

        # pre-sort
        ws_sorted = {m: sorted(self.time_windows[m], key=lambda x: x[0])
                    for m in range(self.real_num_machines)}

        for j in range(self.real_num_jobs):
            rem = float(self.jobs_remaining[j])
            if rem <= eps: 
                continue
            need = min(self.split_min, rem)

            # tra last_end của job j trên từng machine
            last_end_on_m = {}
            jm = self.job_assignments.get(j, {})
            for m, segs in jm.items():
                if segs:
                    # segs đã append theo thời gian; đảm bảo sort một lần nếu cần
                    last_end_on_m[m] = segs[-1][1]  # (start, end)

            for m in range(self.real_num_machines):
                ws = ws_sorted[m]
                L = min(len(ws), self.W_MAX)
                for w in range(L):
                    s, e = ws[w]
                    if (e - s) + eps < need:
                        continue
                    mask[j, m, w] = True
                    # preferred nếu có thể nối ngay (s ≈ last_end)
                    if m in last_end_on_m and abs(s - last_end_on_m[m]) < 1e-6:
                        pref[j, m, w] = True

        if pref.any():
            mask = pref

        return mask.reshape(-1)

    def _max_window_end(self) -> float:
        max_e = 0.0
        for m_ws in self.time_windows.values():
            for s, e in m_ws:
                if e > max_e:
                    max_e = e
        return max_e

    def _decode_machine_windows_from_state(self, state_np: np.ndarray, m: int):
        """Trả list [(start, end)] dài W_MAX (pad 0 nếu thiếu)."""
        base = self.N_MAX + m * (2 * self.W_MAX)
        arr = state_np[base: base + 2 * self.W_MAX]
        return [(float(arr[2*i]), float(arr[2*i+1])) for i in range(self.W_MAX)]

    def _encode_machine_timewindows(self, windows_for_machine):
        """
        Encode một máy thành vector cố định độ dài = 2 * W_MAX.
        Mỗi timewindow lấy (start, end). Nếu nhiều hơn W_MAX -> cắt bớt.
        Nếu ít hơn -> pad (0, 0).
        """
        encoded = []
        # Lấy tối đa W_MAX cửa sổ
        for w in windows_for_machine[: self.W_MAX]:
            if len(w) >= 2:
                # Trường hợp [start, end] hoặc [*, start, end]: lấy 2 phần tử cuối
                start, end = w[-2], w[-1]
            else:
                # Phòng hờ dữ liệu xấu
                start, end = 0, 0
            encoded.extend([start, end])

        # Pad thêm (0,0) nếu thiếu
        missing = self.W_MAX - min(len(windows_for_machine), self.W_MAX)
        if missing > 0:
            encoded.extend([0, 0] * missing)

        # Đảm bảo độ dài đúng
        if len(encoded) != 2 * self.W_MAX:
            encoded = (encoded + [0] * (2 * self.W_MAX - len(encoded)))[: 2 * self.W_MAX]
        return encoded

    def _pad_observation(self) -> np.ndarray:
        """
        Build observation vector với shape:
        obs_dim = N_MAX + M_MAX * (2 * W_MAX)

        - First N_MAX: jobs_remaining (pad 0), chuẩn hoá theo tổng jobs hiện tại
        - Then M_MAX blocks: mỗi block dài 2*W_MAX ứng với (start, end) của windows (pad (0,0)),
        chuẩn hoá theo horizon H = max_end_time hiện tại
        """
        eps = 1e-6
        # Horizon H: lấy max end trong tất cả windows hoặc current_makespan
        H = 0.0
        for m in range(self.real_num_machines):
            if self.time_windows[m]:
                H = max(H, max(e for (s, e) in self.time_windows[m]))
        H = max(H, self.current_makespan, self.lower_bound, 1.0)

        # --- Jobs block (N_MAX) ---
        jobs_sum = sum(self.jobs_remaining) or 1.0
        jobs_vector = (np.array(
            self.jobs_remaining + [0]*(self.N_MAX - self.real_num_jobs),
            dtype=np.float32
        ) / jobs_sum)[: self.N_MAX]

        # --- Machines block (M_MAX * 2*W_MAX) ---
        machine_blocks = []
        for machine_id in range(self.real_num_machines):
            enc = self._encode_machine_timewindows(self.time_windows[machine_id])
            for k, v in enumerate(enc):
                enc[k] = float(v) / H
            machine_blocks.extend(enc)

        # pad thêm các máy trống nếu < M_MAX
        missing_machines = self.M_MAX - self.real_num_machines
        if missing_machines > 0:
            machine_blocks.extend([0.0] * (missing_machines * 2 * self.W_MAX))

        obs = np.concatenate([jobs_vector, np.array(machine_blocks, dtype=np.float32)], axis=0)

        # ép về số chiều lần cuối cho chắc ăn
        expected_len = self.N_MAX + self.M_MAX * (2 * self.W_MAX)
        if obs.shape[0] != expected_len:
            if obs.shape[0] < expected_len:
                obs = np.pad(obs, (0, expected_len - obs.shape[0]), mode="constant")
            else:
                obs = obs[:expected_len]
        return obs

    def state_dim(self) -> int:
        return int(self.observation_space.shape[0])

    def action_dim(self) -> int:
        return int(self.action_space.n)

    def encode_action(self, j: int, m: int, w: int) -> int:
        return j * (self.M_MAX * self.W_MAX) + m * self.W_MAX + w

    def decode_action(self, action_id: int) -> Tuple[int, int, int]:
        j = action_id // (self.M_MAX * self.W_MAX)
        rem = action_id % (self.M_MAX * self.W_MAX)
        m = rem // self.W_MAX
        w = rem % self.W_MAX
        return int(j), int(m), int(w)

    # ----------------- Gym API -----------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        self._fallback_used = False

        self.jobs_remaining = self.processing_times.copy()
        self.machines_finish = [0] * self.real_num_machines
        self.job_assignments = {}
        self.current_makespan = 0
        self.done = False
        self._refresh_earliest_starts()
        return self._pad_observation(), {}

    def _earliest_start_for_machine(self, m: int):
        """
        Trả về start sớm nhất của 1 window trên máy m
        mà có độ dài >= split_min. Nếu không có, trả về None.
        """
        if m >= self.real_num_machines:
            return None
        best = None
        for (s, e) in sorted(self.time_windows[m], key=lambda x: x[0]):
            if (e - s) >= self.split_min:
                best = s
                break
        return best

    def _refresh_earliest_starts(self, machine_id: int | None = None):
        """
        Cập nhật self.earliest_win_start.
        - Nếu machine_id is None: làm lại toàn bộ các máy thật (0..real_num_machines-1)
        - Nếu machine_id là int: chỉ cập nhật máy đó
        Các vị trí > real_num_machines sẽ set = None.
        """
        if self.earliest_win_start is None:
            self.earliest_win_start = [None] * self.M_MAX

        if machine_id is None:
            for m in range(self.real_num_machines):
                self.earliest_win_start[m] = self._earliest_start_for_machine(m)
            # pad phần máy ảo
            for m in range(self.real_num_machines, self.M_MAX):
                self.earliest_win_start[m] = None
        else:
            self.earliest_win_start[machine_id] = self._earliest_start_for_machine(machine_id)

    def get_earliest_starts(self):
        """
        Trả về numpy array dài real_num_machines các start sớm nhất (hoặc np.inf nếu None).
        Có thể dùng để log/quan sát hoặc đưa vào state nếu muốn.
        """
        out = []
        for m in range(self.real_num_machines):
            val = self.earliest_win_start[m]
            out.append(np.inf if val is None else float(val))
        return np.array(out, dtype=np.float32)

    def render(self, mode="human"):
        print(
            f"Jobs Remaining (real): {self.jobs_remaining}, "
            f"Machines (real finish): {self.machines_finish}, "
            f"Assignments: {self.job_assignments}"
        )

    # ----------------- Masking -----------------

    def _sum_capacity_from(self, windows, w0):
        cap = 0.0
        L = min(len(windows), self.W_MAX)
        for w in range(w0, L):
            s, e = windows[w]
            seg = e - s
            if seg >= self.split_min:
                cap += (seg // self.split_min) * self.split_min
        return cap

    def _get_ready_time(self, j: int) -> float:
        """Trả về thời điểm job j sẵn sàng (nếu có field), ngược lại 0."""
        for name in ["ready_time", "ready_times", "job_ready_time", "job_ready_times", "earliest_start"]:
            val = getattr(self, name, None)
            if val is not None:
                try:
                    return float(val[j])
                except Exception:
                    pass
        return 0.0

    def _debug_mask_windows(self, job_id: int, ws_sorted, mask, phase: str):
        """Ghi log debug cho windows được xét khi build mask."""
        import json, time
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "ts": ts,
            "phase": phase,
            "job": int(job_id),
            "current_cmax": float(self.current_makespan),
            "split_min": float(self.split_min),
            "machines": {},
        }
        for m, ws in ws_sorted.items():
            L = min(len(ws), self.W_MAX)
            arr = []
            for w in range(L):
                s, e = ws[w]
                arr.append({
                    "w": w,
                    "start": float(s),
                    "end": float(e),
                    "dur": float(e - s),
                    "mask_on": bool(mask[job_id, m, w])
                })
            record["machines"][int(m)] = arr
        with open("trace_mask.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def valid_action_mask(self) -> np.ndarray:
        """
        Mask (flatten N_MAX*M_MAX*W_MAX):
        - Bật top-K cửa sổ earliest hợp lệ mỗi máy (độ dài ≥ split_min)
        - Ưu tiên các cửa sổ có end <= current Cmax
        - Nếu job đã bind máy -> chỉ bật trên máy đó
        """
        eps = 1e-9
        mask = np.zeros((self.N_MAX, self.M_MAX, self.W_MAX), dtype=bool)
        ws_sorted = {m: sorted(self.time_windows[m], key=lambda x: x[0])
                     for m in range(self.real_num_machines)}

        H = max(self.current_makespan, self.lower_bound, 1.0)
        tol = 1e-6
        K = getattr(self, "topK", 3) or 1
        any_set = False

        for j in range(self.real_num_jobs):
            rem = float(self.jobs_remaining[j])
            if rem <= eps:
                continue
            need = min(float(self.split_min), rem)

            # nếu job đã có máy => chỉ xét máy đó
            bound_m = None
            jm = self.job_assignments.get(j, {})
            if jm:
                for m, segs in jm.items():
                    if segs:
                        bound_m = m
                        break

            for m in range(self.real_num_machines):
                if bound_m is not None and m != bound_m:
                    continue

                ws = ws_sorted[m]
                L = min(len(ws), self.W_MAX)
                if L == 0:
                    continue

                # lọc cửa sổ đủ dài
                cands = [(w, s, e) for w, (s, e) in enumerate(ws[:L])
                         if (e - s) + eps >= need]

                # chia 2 pha: end ≤ Cmax, end > Cmax
                phase1 = [(w, s, e) for (w, s, e) in cands if e <= self.current_makespan + tol]
                phase2 = [(w, s, e) for (w, s, e) in cands if e > self.current_makespan + tol]

                chosen = (phase1[:K] if phase1 else phase2[:K])
                for (w, s, e) in chosen:
                    mask[j, m, w] = True
                    any_set = True

        # fallback: nếu không set gì, bật tất cả cửa sổ đủ dài
        if not any_set:
            for j in range(self.real_num_jobs):
                rem = float(self.jobs_remaining[j])
                if rem <= eps:
                    continue
                need = min(float(self.split_min), rem)
                for m in range(self.real_num_machines):
                    ws = ws_sorted[m]
                    for w, (s, e) in enumerate(ws[:self.W_MAX]):
                        if (e - s) + eps >= need:
                            mask[j, m, w] = True

        # --- debug log ---
        if getattr(self, "verbose", False):
            try:
                for j in range(self.real_num_jobs):
                    self._debug_mask_windows(j, ws_sorted, mask, phase="valid_mask")
            except Exception as ex:
                write_log(f"[debug_mask] error: {ex}", "envs.log")

        return mask.reshape(-1)


    # ----------------- Windows update -----------------
    def _update_windows(self, machine_id: int, start: int, end: int):
        new_ws = []
        for (ws, we) in self.time_windows[machine_id]:
            if we <= start or end <= ws:
                new_ws.append((ws, we)); continue
            if ws < start and end < we:
                if ws < start: new_ws.append((ws, start))
                if end < we:   new_ws.append((end, we))
            elif start <= ws and we <= end:
                pass
            elif ws < start < we <= end:
                new_ws.append((ws, start))
            elif start <= ws < end < we:
                new_ws.append((end, we))
            else:
                if ws < start: new_ws.append((ws, min(start, we)))
                if end < we:   new_ws.append((max(end, ws), we))
        new_ws = [(float(s), float(e)) for (s,e) in new_ws if (e - s) > 1e-9]
        self.time_windows[machine_id] = new_ws
        self.time_windows[machine_id].sort(key=lambda x: x[0])
        self._refresh_earliest_starts(machine_id)

    # ----------------- Transition -----------------
    def step(self, action: int):
        job_id, machine_id, window_id = self.decode_action(action)
        current_total_gap = sum(self.machines_finish)
        # === trace in ===
        if self.verbose:
            import json
            rec0 = {
                "tag": "env_step_in",
                "choose": {"j": int(job_id), "m": int(machine_id), "w": int(window_id)},
                "jobs_remaining_sum": float(sum(self.jobs_remaining)),
                "jobs_remaining_j": float(self.jobs_remaining[job_id]) if job_id < self.real_num_jobs else None,
                "machine_windows_len": int(len(self.time_windows[machine_id])) if machine_id < self.real_num_machines else None
            }
            write_jsonl(json.dumps(rec0, ensure_ascii=False), "trace_steps.jsonl")

        # === validate action ===
        windows = self.time_windows[machine_id] if machine_id < self.real_num_machines else []
        if (
            job_id >= self.real_num_jobs
            or machine_id >= self.real_num_machines
            or self.jobs_remaining[job_id] <= 0
            or window_id >= min(self.W_MAX, len(windows))
        ):
            if self.verbose:
                write_jsonl({"tag": "env_step_out", "reason": "invalid_action"}, "trace_steps.jsonl")
            return self._pad_observation(), -10.0, self.done, False, {"invalid_action": True}

        remaining_time = self.jobs_remaining[job_id]
        windows_subset = windows[window_id:]

        # === plan chunks ===
        if self.mode == "timespan":
            finish_time, chunks = solve_min_timespan_cfg(remaining_time, windows_subset, self.split_min)
        elif self.mode == "leftover":
            finish_time, chunks = solve_feasible_leftover_rule_cfg(remaining_time, windows_subset, self.split_min)
        else:
            finish_time, chunks = assign_job_to_machine(remaining_time, windows_subset, self.split_min)

        if not chunks:
            if self.verbose:
                write_jsonl({"tag": "env_step_out", "reason": "infeasible_no_chunks"}, "trace_steps.jsonl")
            return self._pad_observation(), -10.0, False, False, {"infeasible": True}

        # dùng chunk đầu tiên (như hiện tại)
        _, start, end = chunks[0]
        served = float(end - start)
        if served <= 0.0:
            if self.verbose:
                write_jsonl({"tag": "env_step_out", "reason": "zero_chunk"}, "trace_steps.jsonl")
            return self._pad_observation(), -10.0, False, False, {"zero_chunk": True}

        if finish_time is None:
            if self.verbose:
                write_jsonl({"tag": "env_step_out", "reason": "finish_time_none"}, "trace_steps.jsonl")
            return self._pad_observation(), -10.0, self.done, False, {"infeasible": True}

        write_log(
        f"[update] machine_windows: {self.machines_finish}, j{job_id} m{machine_id} served={served}",
            "machine_windows.log"
        )

        # === snapshot trước khi cập nhật để tính reward ===
        prev_sum = float(sum(self.jobs_remaining))
        old_cmax = float(max(self.machines_finish[:self.real_num_machines], default=0.0))
        prev_finish_m = self.machines_finish[machine_id]

        # === ghi nhận phân công & cập nhật windows/state ===
        self.job_assignments.setdefault(job_id, {}).setdefault(machine_id, []).append((start, end))
        self._update_windows(machine_id, start, end)
        self.jobs_remaining[job_id] = max(0.0, self.jobs_remaining[job_id] - served)
        self.machines_finish[machine_id] = max(prev_finish_m, end)

        write_log(
            f"[update] j{job_id} m{machine_id} served={served} win_m{machine_id}={self.time_windows[machine_id]}",
            "envs.log"
        )

        # === tiến triển? (giảm khối lượng hoặc tăng cmax) ===
        new_makespan = float(max(self.machines_finish[:self.real_num_machines], default=0.0))
        new_sum = float(sum(self.jobs_remaining))
        progress = (abs(new_sum - prev_sum) > 1e-6) or (new_makespan > old_cmax + 1e-6)
        if not progress:
            return self._pad_observation(), -1.0, False, False, {"noprog": True}

        self.current_makespan = new_makespan

        # === chuẩn hóa đơn giản theo mốc H để ổn định thang đo ===
        H = 0.0
        for m in range(self.real_num_machines):
            if self.time_windows[m]:
                H = max(H, max(e for (s, e) in self.time_windows[m]))
        H = max(H, self.current_makespan, self.lower_bound, 1.0)

        # 1) phần thưởng chính: âm của mức tăng Cmax (tăng càng ít càng tốt)
        delta_cmax_norm = (new_makespan - old_cmax) / H
        reward = - float(delta_cmax_norm)

        if self.current_makespan != 0:
            total_gap = (sum(self.machines_finish) - current_total_gap)
            total_gap_norm = np.clip(total_gap / (sum(self.machines_finish) + 1e-6), 1e-6, 10)
        reward = reward + total_gap_norm

        write_log(
            f"[reward_dbg] ΔC={delta_cmax_norm:.4f}, gap={total_gap_norm:.4f}, "
            f"reward={reward:.4f}",
            "rewards.log"
        )

        # 2) phạt rất nặng nếu job này đã/đang nằm trên nhiều máy khác nhau
        #    (không quan tâm số đoạn miễn là CÙNG 1 máy)
        assigned_machines = [m for m in self.job_assignments.get(job_id, {}).keys() if self.job_assignments[job_id].get(m)]
        # nếu đã có ít nhất 1 máy khác với machine_id hiện tại
        cross_penalty = getattr(self, "cross_penalty", 5.0)  # cho phép chỉnh từ ngoài
        if any(m != machine_id for m in assigned_machines[:-1]):  # máy khác ngoài máy hiện tại
            reward = -abs(float(cross_penalty))  # phạt cực nặng
            if self.verbose:
                import json
                write_jsonl(json.dumps({
                    "tag": "env_step_out",
                    "reason": "cross_machine_violation",
                    "job": int(job_id),
                    "machines": assigned_machines,
                    "reward": float(reward)
                }, ensure_ascii=False), "trace_steps.jsonl")
            return self._pad_observation(), float(reward), False, False, {"cross_machine_violation": True}


        write_log(
            f"[update] current_total_gap: {current_total_gap} total_gap: {total_gap}",
            "envs.log"
        )

        # === trace out ===
        if self.verbose:
            import json
            plan_rec = {
                "tag": "plan",
                "served": float(served),
                "chunk": {"start": float(start), "end": float(end)},
                "windows_after_m": int(machine_id),
                "windows_after": [(float(s), float(e)) for (s, e) in self.time_windows[machine_id]],
            }
            write_jsonl(json.dumps(plan_rec, ensure_ascii=False), "trace_steps.jsonl")

            out_rec = {
                "tag": "env_step_out",
                "H": H,
                "reward": float(reward),
                "reward_parts": {
                    "delta_cmax_norm": float(delta_cmax_norm),
                    "cross_machine": int(any(m != machine_id for m in assigned_machines[:-1]))
                },
                "cmax": float(new_makespan),
                "machines_finish": [float(x) for x in self.machines_finish[:self.real_num_machines]],
                "jobs_remaining_sum": float(sum(self.jobs_remaining)),
            }
            write_jsonl(json.dumps(out_rec, ensure_ascii=False), "trace_steps.jsonl")

        # scale/clipping: giữ tín hiệu phạt nặng
        reward = float(np.clip(reward, -10.0, 1.0))

        # === done? ===
        if all(t <= 0 for t in self.jobs_remaining):
            self.done = True
            # nhân reward với timespan cuối cùng để nhấn mạnh Cmax
            final_time_span = max(1.0, float(self.current_makespan))
            reward = float(reward * final_time_span)

            # chuẩn hóa nếu cần (giữ trong [-50, 50] để tránh overflow)
            reward = float(np.clip(reward, -50.0, 50.0))

            if self.verbose:
                write_jsonl({"tag": "env_done", "reason": "all_jobs_completed",
                            "final_time_span": float(final_time_span),
                            "final_reward": float(reward)}, "trace_steps.jsonl")

            return self._pad_observation(), float(reward), True, False, {}

        if not self.valid_action_mask().any():
            if self.verbose:
                write_jsonl({"tag": "env_done", "reason": "no_actions"}, "trace_steps.jsonl")
            return self._pad_observation(), float(reward), True, False, {"no_actions": True}

        return self._pad_observation(), float(reward), False, False, {}
