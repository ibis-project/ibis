#!/bin/bash -e

function run()
{
    local python_version="${1}"
    local no_dots_python_version="${python_version//.}"
    pyenv install --skip-existing "${python_version}"
    pyenv local "${python_version}"
    tox -e "py${no_dots_python_version:0:2}"
}

run "$@"
