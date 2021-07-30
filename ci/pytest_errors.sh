#!/bin/bash -e
# parse the pytest log and print the pytest errors in a concise way

if [[ -f "pytest.log" ]]; then
    csplit pytest.log "/= FAILURES =/" "/= short test summary info =/" '{*}' > /dev/null
    mv xx00 pytest_progress.log
    mv xx01 pytest_errors.log
    mv xx02 pytest_summary.log

    head -n1 pytest_summary.log >> pytest_errors.log
    grep "^FAILED \|^ERROR " pytest_summary.log >> pytest_errors.log
    tail -n1 pytest_summary.log >> pytest_errors.log

    cat pytest_errors.log
fi
