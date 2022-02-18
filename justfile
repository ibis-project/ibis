# list justfile recipes
default:
    just --list

# clean untracked files
clean:
    git clean -fdx -e 'ci/ibis-testing-data'

# show all backends
@list-backends:
    yj -tj < pyproject.toml | \
        jq -rcM '.tool.poetry.plugins["ibis.backends"] | keys[]' | grep -v '^spark' | sort

# format code
fmt:
    black .
    isort .
    pyupgrade --py38-plus ibis/**/*.py

# run all non-backend tests in parallel; additional arguments are forwarded to pytest
check *args:
    pytest -m 'not backend' -q -n auto {{ args }}

# run pytest for ci; additional arguments are forwarded to pytest
ci-check *args:
    poetry run pytest --durations=25 -ra --junitxml=junit.xml --cov=ibis --cov-report=xml:coverage.xml {{ args }}

# lint code
lint:
    black -q . --check
    isort -q . --check
    flake8 .

# type check code using mypy
typecheck:
    mypy .

# run the test suite for one or more backends
test +backends:
    #!/usr/bin/env bash
    set -euo pipefail

    pytest_args=("-m" "$(sed 's/ / or /g' <<< '{{ backends }}')")

    if ! [[ "{{ backends }}" =~ impala|pyspark ]]; then
        pytest_args+=("-n" "auto" "-q")
    fi

    pytest "${pytest_args[@]}"

# download testing data
download-data owner="ibis-project" repo="testing-data" rev="master":
    #!/usr/bin/env bash
    outdir="{{ justfile_directory() }}/ci/ibis-testing-data"
    rm -rf "$outdir"
    url="https://github.com/{{ owner }}/{{ repo }}"

    args=("$url")
    if [ "{{ rev }}" = "master" ]; then
        args+=("--depth" "1")
    fi
    args+=("$outdir")
    git clone "${args[@]}"

# start backends using docker compose and load data; no arguments starts all backends
up *backends:
    docker compose up --wait {{ backends }}
    just load-data {{ backends }}

# stop and remove containers; clean up networks and volumes
down:
    docker compose down --volumes --remove-orphans

# load data into running backends; requires a running backend
load-data *backends="all":
    python ci/datamgr.py -v load {{ backends }}
