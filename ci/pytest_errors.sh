#!/bin/bash -e
# parse the pytest log and print the pytest errors in a concise way

csplit pytest.log "/short test summary info/" '{*}' > /dev/null
mv xx00 pytest_errors.log
mv xx01 pytest_summary.log

head -n1 pytest_summary.log >> pytest_errors.log
grep "^FAILED \|^ERROR " pytest_summary.log >> pytest_errors.log
tail -n1 pytest_summary.log >> pytest_errors.log

cat pytest_errors.log
