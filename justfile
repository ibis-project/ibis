# list justfile recipes
default:
    just --list

# clean untracked files
clean:
    git clean -fdx -e 'ci/ibis-testing-data'

# verify poetry version
check-poetry:
    #!/usr/bin/env bash
    set -euo pipefail

    required_version="1.8.3"
    version="$(poetry --version)"
    if ! grep -qF "${required_version}" <<< "${version}"; then
        >&2 echo "poetry version must be ${required_version}, got ${version}"
        exit 1
    fi

# lock dependencies without updating existing versions
lock: check-poetry
    #!/usr/bin/env bash
    set -euo pipefail

    poetry lock --no-update
    just export-deps

# update locked dependencies
update *deps: check-poetry
    #!/usr/bin/env bash
    set -euo pipefail

    poetry update --lock {{ deps }}
    just export-deps

# export locked dependencies
export-deps:
    #!/usr/bin/env bash
    set -euo pipefail

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

# run backend doctests
backend-doctests backend *args:
    #!/usr/bin/env bash
    args=(pytest --doctest-modules {{ args }})
    for file in ibis/backends/{{ backend }}/**.py; do
        if grep -qPv '.*test.+' <<< "${file}"; then
            args+=("${file}")
        fi
    done
    if [ -n "${CI}" ]; then
        poetry run "${args[@]}"
    else
        "${args[@]}"
    fi

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

# run doctests
doctest *args:
    #!/usr/bin/env bash
    set -eo pipefail

    if [ -n "${CI}" ]; then
        runner=(poetry run)
    else
        runner=(python -m)
    fi

    # TODO(cpcloud): why doesn't pytest --ignore-glob=test_*.py work?
    "${runner[@]}" pytest --doctest-modules {{ args }} $(
      find \
        ibis \
        -wholename '*.py' \
        -and -not -wholename '*test*.py' \
        -and -not -wholename '*__init__*' \
        -and -not -wholename '*gen_*.py' \
        -and -not -wholename '*ibis/backends/flink/*' # FIXME(deepyaman)
    )

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

# download the iceberg jar used for testing pyspark and iceberg integration
download-iceberg-jar pyspark scala="2.12" iceberg="1.6.1":
    #!/usr/bin/env bash
    set -eo pipefail

    runner=(python)

    if [ -n "${CI}" ]; then
        runner=(poetry run python)
    fi
    pyspark="$("${runner[@]}" -c "import pyspark; print(pyspark.__file__.rsplit('/', 1)[0])")"
    pushd "${pyspark}/jars"
    jar="iceberg-spark-runtime-{{ pyspark }}_{{ scala }}-{{ iceberg }}.jar"
    url="https://search.maven.org/remotecontent?filepath=org/apache/iceberg/iceberg-spark-runtime-{{ pyspark }}_{{ scala }}/{{ iceberg }}/${jar}"
    curl -qSsL -o "${jar}" "${url}"
    ls "${jar}"

# start backends using docker compose; no arguments starts all backends
up *backends:
    #!/usr/bin/env bash
    set -eo pipefail

    if [ -n "$CI" ]; then
        # don't show a big pile of output when running in CI
        args=(--quiet-pull --no-color)
    else
        args=()
    fi

    docker compose up --build --wait "${args[@]}" {{ backends }}

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

# stop all containers, prune networks, and remove all volumes
stop *backends:
    just down {{ backends }}
    docker network prune -f
    docker volume prune -af

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

# profile something
profile +args:
    pyinstrument {{ args }}

# generate API documentation
docs-apigen *args:
    cd docs && quartodoc interlinks
    quartodoc build {{ args }} --config docs/_quarto.yml

# build documentation
docs-render:
    #!/usr/bin/env bash
    set -euo pipefail

    # Check if the folder "reference" exists and has contents
    if [ ! -d "docs/reference" ] || [ -z "$(ls -A docs/reference)" ]; then
        just docs-apigen
    fi

    quarto render docs

# preview docs
docs-preview:
    #!/usr/bin/env bash
    set -euo pipefail

    # Check if the folder "reference" exists and has contents
    if [ ! -d "docs/reference" ] || [ -z "$(ls -A docs/reference)" ]; then
        just docs-apigen
    fi

    quarto preview docs

# regen api and preview docs
docs-api-preview:
    just docs-apigen --verbose
    quarto preview docs

# deploy docs to netlify
docs-deploy:
    quarto publish --no-prompt --no-browser --no-render netlify docs

# build jupyterlite repl
build-jupyterlite:
    #!/usr/bin/env bash
    set -euo pipefail

    mkdir -p docs/_output/jupyterlite

    rm -rf dist/
    poetry-dynamic-versioning
    ibis_dev_version="$(poetry version | cut -d ' ' -f2)"
    poetry build --format wheel
    git checkout pyproject.toml ibis/__init__.py

    jupyter lite build \
        --debug \
        --no-libarchive \
        --piplite-wheels "dist/ibis_framework-${ibis_dev_version}-py3-none-any.whl" \
        --piplite-wheels "https://duckdb.github.io/duckdb-pyodide/wheels/duckdb-1.1.0-cp311-cp311-emscripten_3_1_46_wasm32.whl" \
        --apps repl \
        --no-unused-shared-packages \
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
