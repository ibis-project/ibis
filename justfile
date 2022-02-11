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
    poetry run black .
    poetry run isort .
    poetry run pyupgrade --py38-plus ibis/**/*.py

# run all non-backend tests in parallel; additional arguments are forwarded to pytest
check *args:
    poetry run pytest -m 'not backend' -q -n auto {{ args }}

# run pytest for ci; additional arguments are forwarded to pytest
ci-check *args:
    poetry run pytest --durations=25 -ra --junitxml=junit.xml --cov=ibis --cov-report=xml:coverage.xml {{ args }}

# lint code
lint:
    poetry run black -q . --check
    poetry run isort -q . --check
    poetry run flake8 .

# type check code using mypy
typecheck:
    poetry run mypy .

# run the test suite for one or more backends
test +backends:
    #!/usr/bin/env bash
    set -euo pipefail

    pytest_args=("-m" "$(sed 's/ / or /g' <<< '{{ backends }}')")

    if ! [[ "{{ backends }}" =~ impala|pyspark ]]; then
        pytest_args+=("-n" "auto" "-q")
    fi

    poetry run pytest "${pytest_args[@]}"

# download testing data
download-data owner="ibis-project" repo="testing-data" rev="master":
    #!/usr/bin/env bash
    set -euo pipefail

    outdir="{{ justfile_directory() }}/ci/ibis-testing-data"
    rm -rf "$outdir"
    tmpdatadir="$(mktemp -d)"
    tmpdir="$(mktemp -d)"
    curl -L -o "$tmpdatadir/data.zip" "https://github.com/{{ owner }}/{{ repo }}/archive/{{ rev }}.zip"
    unzip "$tmpdatadir/data.zip" -d "$tmpdir"
    mv "$tmpdir/{{ repo }}-{{ rev }}" "$outdir"
    rm -r "$tmpdir" "$tmpdatadir"

# start backends using docker compose and load data; no arguments starts all backends
up *backends:
    docker compose up --wait {{ backends }}
    just load-data {{ backends }}

# stop and remove containers; clean up networks and volumes
down:
    docker compose down --volumes --remove-orphans

# load data into running backends; requires a running backend
load-data *backends:
    #!/usr/bin/env bash
    set -euo pipefail

    if [ -z "{{ backends }}" ]; then
        mapfile -t services < <(just list-backends | grep -v '^spark')
    else
        mapfile -t -d ' ' services <<< "{{ backends }}"
    fi
    for backend in "${services[@]}"; do
        poetry run python ci/datamgr.py "$(tr -d '\n' <<< "${backend}")" &
    done

    wait
