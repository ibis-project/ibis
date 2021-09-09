#!/usr/bin/env bash

set -euo pipefail

# parse the pytest log and print the pytest errors in a concise way

if [[ -f "pytest.log" ]]; then
  if csplit pytest.log "/= FAILURES =/" "/= short test summary info =/" '{*}' > /dev/null; then
    mv xx00 pytest_progress.log
    mv xx01 pytest_errors.log
    mv xx02 pytest_summary.log

    cat pytest_errors.log <({
      head -n1 pytest_summary.log
      grep "^FAILED \|^ERROR " pytest_summary.log
      tail -n1 pytest_summary.log
    })
  fi
fi
