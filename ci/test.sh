#!/bin/bash -e

docker-compose build --pull ibis && \
    docker-compose run ibis \
	bash -c \
	    "find /ibis -name '*.py[co]' -delete && \
		rm -rf $(find /ibis -type d -name '__pycache__') && \
		pytest $@"
