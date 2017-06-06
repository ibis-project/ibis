#!/bin/bash -e

pyenv install --skip-existing "${1}"
pyenv local "${1}"
tox -e "py${${1//.}:0:2}"
