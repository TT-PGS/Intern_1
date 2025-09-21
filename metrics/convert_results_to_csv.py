"""
Convert scheduling result JSON files into two aggregate CSVs:
1) summary.csv    - one row per (file, algorithm)
2) assignments.csv- one row per assignment (job row)

Usage:
    python convert_results_to_csv.py \
        --input-dir ./results_json \
        --summary-csv summary.csv \
        --assignments-csv assignments.csv
"""
import argparse
import csv
import json
import os
from glob import glob
from typing import Any, Dict, Iterable, List, Optional

ALGOKEYS = ["fcfs_results", "sa_results", "ga_results"]

def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse JSON: {path} ({e})")
        return None

def ensure_list_of_lists(x: Any) -> Optional[List[List[int]]]:
    """Ensure segments/plan to be a JSON-serializable list-of-lists (or None)."""
    if x is None:
        return None
    if isinstance(x, list):
        # Shallow verify elements are list-like
        return x
    return None

def to_json_str(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return ""

def extract_summary_rows(doc: Dict[str, Any], src_file: str) -> Iterable[List[Any]]:
    dataset = doc.get("dataset")
    lower_bound = doc.get("lower_bound")
    split_mode = doc.get("split_mode")
    seed = doc.get("seed_used_on_sa_and_ga")

    for algokey in ALGOKEYS:
        if algokey not in doc:
            continue
        res = doc[algokey] or {}
        algo_name = algokey.replace("_results", "").upper()
        makespan = res.get("makespan")
        percentage_gap = res.get("percentage_gap")
        processing_time_ms = res.get("processing_time_milliseconds")

        # Optional: GA best genome info, SA best order
        extra = {}
        if algokey == "ga_results":
            best_genome = res.get("best_genome", {})
            if best_genome:
                extra["ga_order"] = best_genome.get("order")
                extra["ga_machine_genes"] = best_genome.get("machine_genes")
        elif algokey == "sa_results":
            extra["sa_best_order"] = res.get("best_order")

        yield [
            os.path.basename(src_file),
            dataset,
            lower_bound,
            split_mode,
            algo_name,
            makespan,
            percentage_gap,
            processing_time_ms,
            seed if algo_name != "FCFS" else None,
            to_json_str(extra) if extra else ""
        ]

def extract_assignment_rows(doc: Dict[str, Any], src_file: str) -> Iterable[List[Any]]:
    dataset = doc.get("dataset")
    for algokey in ALGOKEYS:
        if algokey not in doc:
            continue
        res = doc[algokey] or {}
        algo_name = algokey.replace("_results", "").upper()
        assignments = res.get("assignments", []) or []
        for a in assignments:
            job = a.get("job")
            machine = a.get("machine")
            fragments = a.get("fragments")
            timespan = a.get("timespan")
            finish = a.get("finish")

            # FCFS uses "plan", SA/GA use "segments"
            seg = a.get("segments", None)
            plan = a.get("plan", None)
            segments_or_plan = ensure_list_of_lists(seg if seg is not None else plan)
            seg_json = to_json_str(segments_or_plan)

            yield [
                os.path.basename(src_file),
                dataset,
                algo_name,
                job,
                machine,
                fragments,
                timespan,
                finish,
                seg_json
            ]

def main():
    parser = argparse.ArgumentParser(description="Batch-convert result JSON files to aggregate CSVs.")
    parser.add_argument("--input-dir", type=str, default=".", help="Directory containing JSON files")
    parser.add_argument("--summary-csv", type=str, default="summary.csv", help="Output CSV for high-level summaries")
    parser.add_argument("--assignments-csv", type=str, default="assignments.csv", help="Output CSV for per-assignment rows")
    parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern for JSON files")
    args = parser.parse_args()

    json_files = sorted(glob(os.path.join(args.input_dir, args.pattern)))
    if not json_files:
        print(f"[INFO] No JSON files found in: {args.input_dir} with pattern: {args.pattern}")
        return

    # Prepare CSV writers
    summary_header = [
        "source_file",
        "dataset",
        "lower_bound",
        "split_mode",
        "algo",
        "makespan",
        "percentage_gap",
        "processing_time_ms",
        "seed",
        "extra"
    ]
    assignments_header = [
        "source_file",
        "dataset",
        "algo",
        "job",
        "machine",
        "fragments",
        "timespan",
        "finish",
        "segments_or_plan_json"
    ]

    total_docs = 0
    total_summary_rows = 0
    total_assignment_rows = 0

    with open(args.summary_csv, "w", newline="", encoding="utf-8") as fs, \
         open(args.assignments_csv, "w", newline="", encoding="utf-8") as fa:
        summary_writer = csv.writer(fs)
        assignment_writer = csv.writer(fa)
        summary_writer.writerow(summary_header)
        assignment_writer.writerow(assignments_header)

        for jf in json_files:
            doc = read_json(jf)
            if not doc or not isinstance(doc, dict):
                continue
            total_docs += 1

            for row in extract_summary_rows(doc, jf):
                summary_writer.writerow(row)
                total_summary_rows += 1

            for row in extract_assignment_rows(doc, jf):
                assignment_writer.writerow(row)
                total_assignment_rows += 1

    print(f"[DONE] Parsed {total_docs} JSON file(s).")
    print(f"[DONE] Wrote {total_summary_rows} row(s) to {args.summary_csv}.")
    print(f"[DONE] Wrote {total_assignment_rows} row(s) to {args.assignments_csv}.")

if __name__ == "__main__":
    main()
