#!/bin/bash
#
# Run the full local test suite (lint, unit tests with --runslow, smoke tests)
# and write every report into reports/, which is gitignored. CI badges come from
# GitHub Actions, so nothing here is meant to be committed.

set -e

echo "Running flake8"
flake_test=`flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
if [[ "${flake_test}" != 0 ]]; then
    echo "${flake_test}"
    echo "flake failed"
    exit 1
fi

rm -rf reports
mkdir -p reports/junit reports/coverage reports/badges

echo "Running flake8, aggressive"
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics > reports/flake.txt

echo "Running coverage.py"
coverage erase
NUMBA_DISABLE_JIT=1 coverage run --branch -m pytest tests/tfscreen --runslow --junit-xml=reports/junit/junit.xml
NUMBA_DISABLE_JIT=1 pytest tests/smoke-tests --runslow

echo "Generating reports"
coverage html -d reports/htmlcov
coverage xml -o reports/coverage/coverage.xml

genbadge tests -o reports/badges/tests-badge.svg
genbadge coverage -o reports/badges/coverage-badge.svg

echo "Reports written to reports/"
