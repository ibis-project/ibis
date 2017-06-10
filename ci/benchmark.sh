#!/bin/bash -e
function benchmark()
{
    which asv
    # make existing data available
    mkdir -p .asv

    # clone existing benchmark data
    pushd ..
    git clone git@github.com:ibis-project/ibis-benchmarks
    [ -e "asv/results" ] && mv asv/results ../ibis/.asv
    [ -e "asv/html " ] && mv asv/html ../ibis/.asv
    popd

    # run an asv command
    local -a params=(${2})
    PATH="$HOME/miniconda/bin:$PATH" ${1} ${params[@]}

    if [ "${3}" = "--publish" ]; then
	# generate html
	asv publish

	[ -e ".asv/html/graphs" ] || exit 1

	# remove conda environments
	rm -rf .asv/env

	# replace old benchmarks with new data and html content
	rm -rf ../ibis-benchmarks/asv
	mv .asv ../ibis-benchmarks/asv

	# push to gh
	git config --global user.name "Circle CI Benchmark"
	git config --global user.email ""

	pushd ../ibis-benchmarks
	echo
	ls -1 asv/results/circle/*.json
	echo
	git add --verbose --ignore-removal .
	git commit --message="Update benchmarks ibis-project/ibis@${CIRCLE_SHA1:-$(git rev-parse HEAD)}"
	git push origin master
	popd
    fi
}

benchmark "$@"
