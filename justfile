# list justfile recipes
default:
    just --list

# clean untracked files
clean:
    git clean -fdx -e 'ci/ibis-testing-data'

# install dependencies for a given backend, or all dependencies if none is given
sync backend="":
    #!/usr/bin/env bash
    if [ ! "{{ backend }}" ]; then
        uv sync --all-groups --all-extras
    else
        uv sync --group dev --group tests --extra {{ backend }} --extra examples --extra geospatial
    fi

# lock dependencies without updating existing versions
lock:
    #!/usr/bin/env bash
    set -euo pipefail

    uv sync --all-extras --group dev --group tests --group docs --no-install-project --no-install-workspace
    just export-deps > requirements-dev.txt

# update locked dependencies
update *packages:
    #!/usr/bin/env bash
    set -euo pipefail

    packages=({{ packages }})
    args=(--all-extras --group dev --group tests --group docs --no-install-project --no-install-workspace)

    if [ "${#packages[@]}" -eq 0 ]; then
        args+=(--upgrade)
    else
        for package in "${packages[@]}"; do
            args+=(--upgrade-package "${package}")
        done
    fi

    uv sync "${args[@]}"

    just export-deps > requirements-dev.txt

# export locked dependencies
@export-deps:
    uv export \
        --frozen \
        --format requirements-txt \
        --all-extras \
        --group dev \
        --group tests \
        --group docs \
        --no-hashes \
        --no-header

# show all backends
@list-backends:
    yj -tj < pyproject.toml | jq -rcM '.project["entry-points"]["ibis.backends"] | keys | sort[]'

# format code
@fmt:
    ruff format --quiet .
    ruff check --quiet --fix .

# run all non-backend tests; additional arguments are forwarded to pytest
check *args:
    pytest -m core {{ args }}

# run pytest for ci; additional arguments are forwarded to pytest
ci-check extras *args:
    uv run --group tests {{ extras }} pytest --junitxml=junit.xml --cov=ibis --cov-report=xml:coverage.xml {{ args }}

# run backend doctests
backend-doctests backend *args:
    #!/usr/bin/env bash
    args=()

    if [ -n "${CI}" ]; then
        args=(uv run --extra {{ backend }} --group tests)
    fi

    args+=(pytest --doctest-modules {{ args }})
    args+=($(find ibis/backends/{{ backend }} -name '*.py' -not -wholename '*test*.py'))

    "${args[@]}"

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
        runner=(uv run --all-extras --group tests)
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
        runner=(uv run --extra pyspark python)
    fi
    pyspark="$("${runner[@]}" -c "import pyspark, os; print(pyspark.__file__.rsplit(os.sep, 1)[0])")"
    pushd "${pyspark}/jars"
    jar="iceberg-spark-runtime-{{ pyspark }}_{{ scala }}-{{ iceberg }}.jar"
    url="https://search.maven.org/remotecontent?filepath=org/apache/iceberg/iceberg-spark-runtime-{{ pyspark }}_{{ scala }}/{{ iceberg }}/${jar}"
    curl -qSsL -o "${jar}" "${url}"
    ls "${jar}"

# pull images
pull *backends:
    #!/usr/bin/env bash
    set -eo pipefail

    backends=({{ backends }})
    buildable=()
    pullable=()

    for backend in "${backends[@]}"; do
        if [ "${backend}" = "flink" -o "${backend}" = "postgres" ]; then
            buildable+=("${backend}")
        else
            pullable+=("${backend}")
        fi
    done

    if [ "${#backends[@]}" -eq 0 ]; then
        docker compose pull --ignore-buildable
        docker compose build "${buildable[@]}" --pull
    elif [ "${#buildable[@]}" -gt 0 ]; then
        docker compose build "${buildable[@]}" --pull
    elif [ "${#pullable[@]}" -gt 0 ]; then
        docker compose pull "${pullable[@]}" --ignore-buildable
    fi

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
    docker system prune --force --volumes
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

    ibis_dev_version="$(just bump-version)"
    uv build --wheel

    git checkout pyproject.toml ibis/__init__.py uv.lock

    jq '{"PipliteAddon": {"piplite_urls": [$ibis]}}' -nM \
        --arg ibis "dist/ibis_framework-${ibis_dev_version}-py3-none-any.whl" \
        > jupyter_lite_config.json

    curl -SLsO https://storage.googleapis.com/ibis-tutorial-data/penguins.csv
    jupyter lite build \
        --debug \
        --contents penguins.csv  \
        --no-libarchive \
        --apps repl \
        --no-unused-shared-packages \
        --output-dir docs/_output/jupyterlite
    rm -f penguins.csv
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

# compute the next version number
@compute-version:
    uv run --only-group dev python ci/release/bump_version.py

# bump the version number in necessary files
bump-version:
    #!/usr/bin/env bash

    ibis_dev_version="$(just compute-version)"
    uvx --from=toml-cli toml set --toml-path=pyproject.toml project.version "$ibis_dev_version" > /dev/null
    sed -i 's/__version__ = .\+/__version__ = "'$ibis_dev_version'"/g' ibis/__init__.py
    just lock > /dev/null
    echo "$ibis_dev_version"
