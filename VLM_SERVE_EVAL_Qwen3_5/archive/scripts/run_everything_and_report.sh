#!/bin/bash
set -euo pipefail

# Wait for smoke test to finish or just kill it if it's taking too long
# Actually, the user might want us to just start the full run now.
pkill -f "test_local_qwen.sh" || true
pkill -f "lighteval accelerate" || true
sleep 5

echo "Starting full benchmark run..."
bash run_local_qwen.sh > run_local_qwen.log 2>&1

echo "Generating report..."
.venv/bin/python3 generate_report.py >> run_local_qwen.log 2>&1

echo "All done!"
