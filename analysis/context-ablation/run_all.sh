#!/usr/bin/env bash
# Rebuild every table in ABLATION_REPORT.md from bench_runs/, in order.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$here"
python3 collect.py
python3 collect_legs.py
{
    python3 analyze.py
    printf '\n\n########## growth.py\n\n';    python3 growth.py
    printf '\n\n########## logscan.py\n\n';   python3 logscan.py
    printf '\n\n########## findings.py\n\n';  python3 findings.py
    printf '\n\n########## examples.py\n\n'; python3 examples.py
    printf '\n\n########## censoring.py\n\n'; python3 censoring.py
} | tee results.txt
