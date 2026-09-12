#!/usr/bin/env python3
"""Recompute the table in `judge_error_needs_a_hard_judging_task_2026-09-12.md`.

Offline: it reads the committed audit stores and nothing else. Exists because
the first draft of that table quoted BBH's solver accuracy as 67.3%, which is
that workload's judge *agreement* rate -- two numbers that live next to each
other in every analysis and mean opposite things.

    python reports/judge_error_table.py
"""

import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.audit_judge_repair import load_records  # noqa: E402

STORES = [
    ("HotpotQA", "reports/audit_phase0_2026-09-09.jsonl"),
    ("BBH", "reports/audit_phase0_bbh_2026-09-10.jsonl"),
    ("GSM8K", "reports/audit_phase0_2026-09-12_gsm8k.jsonl"),
    ("GSM-Hard", "reports/audit_phase0_2026-09-12_gsm_hard.jsonl"),
    ("MBPP", "reports/audit_phase0_2026-09-12_mbpp.jsonl"),
]


def main() -> None:
    print(f"{'workload':10} {'n':>4} {'solver right':>13} {'disagreement':>13}")
    for label, path in STORES:
        records = load_records(pathlib.Path(path))
        if not records:
            print(f"{label:10} {'--':>4}  (no resolved pairs)")
            continue
        right = sum(r.oracle_score for r in records) / len(records)
        differ = sum(1 for r in records
                     if r.verifier_score != r.oracle_score) / len(records)
        print(f"{label:10} {len(records):>4} {right:>12.1%} {differ:>13.1%}")


if __name__ == "__main__":
    main()
