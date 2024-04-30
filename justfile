# list justfile recipes
default:
    just --list

# clean untracked files
clean:
    git clean -fdx -e 'ci/ibis-testing-data'

# lock dependencies without updating existing versions
lock:
    #!/usr/bin/env bash
    set -euo pipefail

    required_version="1.8.2"
    version="$(poetry --version)"
    if ! grep -qF "${required_version}" <<< "${version}"; then
        >&2 echo "poetry version must be ${required_version}, got ${version}"
        exit 1
    fi
    poetry lock --no-update
    poetry export --all-extras --with dev --with test --with docs --without-hashes --no-ansi > requirements-dev.txt

# show all backends
@list-backends:
    yj -tj < pyproject.toml | \
        jq -rcM '.tool.poetry.plugins["ibis.backends"] | keys[]' | grep -v '^spark' | sort

# format code
fmt:
    ruff format .
    ruff check --fix .

# run all non-backend tests; additional arguments are forwarded to pytest
check *args:
    pytest -m core {{ args }}

# run pytest for ci; additional arguments are forwarded to pytest
ci-check *args:
    poetry run pytest --junitxml=junit.xml --cov=ibis --cov-report=xml:coverage.xml {{ args }}

# lint code
lint:
    ruff format -q . --check
    ruff check .

# run the test suite for one or more backends
test +backends:
    #!/usr/bin/env bash
    set -euo pipefail

    pytest_args=("-m" "$(sed 's/ / or /g' <<< '{{ backends }}')")

    if ! [[ "{{ backends }}" =~ impala|pyspark ]]; then
        pytest_args+=("-n" "auto" "-q" "--dist" "loadgroup")
    fi

    pytest "${pytest_args[@]}"

_doctest runner *args:
    #!/usr/bin/env bash
    set -euo pipefail

    # TODO(cpcloud): why doesn't pytest --ignore-glob=test_*.py work?
    {{ runner }} pytest --doctest-modules {{ args }} $(
      find \
        ibis \
        -wholename '*.py' \
        -and -not -wholename '*test*.py' \
        -and -not -wholename '*__init__*' \
        -and -not -wholename '*gen_*.py' \
        -and -not -wholename '*ibis/expr/selectors.py' \
        -and -not -wholename '*ibis/backends/flink/*' # FIXME(deepyaman)
    )

# run doctests
doctest *args:
    just _doctest "python -m" {{ args }}

# run doctests using poetry
ci-doctest *args:
    just _doctest "poetry run" {{ args }}

# download testing data
download-data owner="ibis-project" repo="testing-data" rev="master":
    #!/usr/bin/env bash
    set -euo pipefail

    outdir="{{ justfile_directory() }}/ci/ibis-testing-data"
    rm -rf "$outdir"
    url="https://github.com/{{ owner }}/{{ repo }}"

    args=("$url")
    if [ "{{ rev }}" = "master" ]; then
        args+=("--depth" "1")
    fi

    args+=("$outdir")
    git clone "${args[@]}"

    if [ "{{ rev }}" != "master" ]; then
        git -C "${outdir}" checkout "{{ rev }}"
    fi

# start backends using docker compose; no arguments starts all backends
up *backends:
    docker compose up --build --wait {{ backends }}

# stop and remove containers -> clean up dangling volumes -> start backends
reup *backends:
    just down {{ backends }}
    docker system prune --force --volumes
    just up {{ backends }}

# stop and remove containers; clean up networks and volumes
down *backends:
    #!/usr/bin/env bash
    set -euo pipefail

    if [ -z "{{ backends }}" ]; then
        docker compose down --volumes --remove-orphans
    else
        docker compose rm {{ backends }} --force --stop --volumes
    fi

# tail logs for one or more services
tail *services:
    docker compose logs --follow {{ services }}

# run the benchmark suite
bench +args='ibis/tests/benchmarks':
    pytest --benchmark-only --benchmark-enable --benchmark-autosave {{ args }}

# run benchmarks and compare with a previous run
benchcmp number *args:
    just bench --benchmark-compare {{ number }} {{ args }}

# check for invalid links in a locally built version of the docs
checklinks *args:
    #!/usr/bin/env bash
    set -euo pipefail

    lychee --base docs/_output $(find docs/_output -name '*.html') {{ args }}

# view the changelog for upcoming release (use --pretty to format with glow)
view-changelog flags="":
    #!/usr/bin/env bash
    set -euo pipefail

    npx -y -p conventional-changelog-cli \
        -- conventional-changelog --config ./.conventionalcommits.js \
        | ([ "{{ flags }}" = "--pretty" ] && glow -p - || cat -)

# run the decouple script to check for prohibited inter-module dependencies
decouple +args:
    python ci/check_disallowed_imports.py {{ args }}

# profile something
profile +args:
    pyinstrument {{ args }}

# generate API documentation
docs-apigen *args:
    cd docs && quartodoc interlinks
    quartodoc build {{ args }} --config docs/_quarto.yml

# build documentation
docs-render:
    quarto render docs

# preview docs
docs-preview:
    quarto preview docs

# regen api and preview docs
docs-api-preview:
    just docs-apigen --verbose
    quarto preview docs

# deploy docs to netlify
docs-deploy:
    quarto publish --no-prompt --no-browser --no-render netlify docs

# build an ibis_framework wheel that works with pyodide
build-ibis-for-pyodide:
    #!/usr/bin/env bash
    set -euo pipefail

    # TODO(cpcloud): remove when:
    # 1. pyarrow release contains pyodide
    # 2. ibis supports this version of pyarrow
    rm -rf dist/
    poetry add 'pyarrow>=10.0.1' --allow-prereleases
    poetry build --format wheel
    git checkout poetry.lock pyproject.toml
    jq '{"PipliteAddon": {"piplite_urls": [$ibis, $duckdb]}}' -nM \
        --arg ibis dist/*.whl \
        --arg duckdb "https://duckdb.github.io/duckdb-pyodide/wheels/duckdb-0.10.2-cp311-cp311-emscripten_3_1_46_wasm32.whl" \
        > docs/jupyter_lite_config.json

# build the jupyterlite deployment
build-jupyterlite: build-ibis-for-pyodide
    #!/usr/bin/env bash
    set -euo pipefail

    mkdir -p docs/_output/jupyterlite
    jupyter lite build \
        --debug \
        --no-libarchive \
        --config docs/jupyter_lite_config.json \
        --output-dir docs/_output/jupyterlite
    # jupyter lite build can copy from the nix store, and preserves the
    # original write bit; without this the next run of this rule will result in
    # a permission error when the build tries to remove existing files
    chmod -R u+w docs/_output/jupyterlite

# run the entire docs build pipeline
docs-build-all:
    just docs-apigen --verbose
    just docs-render
    just build-jupyterlite
    just checklinks docs/_output --offline --no-progress

# open chat
chat *args:
    zulip-term {{ args }}
