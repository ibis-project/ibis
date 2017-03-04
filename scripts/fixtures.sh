#!/bin/bash

grep -o --no-filename -P -R '(?<=pytest\.mark\.)([^(\s]+)' "${1}" | \
  sort -u | grep -v '^$' | grep -P -v 'parametrize|xfail|skipif'
