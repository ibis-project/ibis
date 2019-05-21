#!/usr/bin/env bash

CWD=$(dirname $0)

pip install asv
$CWD/asvconfig.py $1 | tee $HOME/.asv-machine.json
git remote add upstream https://github.com/ibis-project/ibis
git fetch upstream refs/heads/master
asv continuous -f 1.5 -e upstream/master $2 || echo > /dev/null
