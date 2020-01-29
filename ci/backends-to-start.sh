#!/bin/bash

# Among the backends, the script finds those that will need to be launched
# by the `docker-compose`. The choice depends on comparing each backend with
# elements inside the SERVICES variable.
#
# Usage:
#  $ ./ci/backends-to-start.sh param1 param2
# * param1: string of backends
# * param2: string of backends, which then need to launched by `docker-compose`
#     (as docker's services) before working with them
#
# Example:
# current_backends=`./ci/backends-to-start.sh "omniscidb impala parquet" "omniscidb impala"`  && echo $current_backends
#
# Output:
# 'omniscidb impala'

# convert strings to arrays
BACKENDS=($(echo $1))
SERVICES=($(echo $2))

# lookup table to choose backends to start
declare -A SERVICES_START
for service in ${SERVICES[@]}
do
    SERVICES_START[$service]=1
done

i=0
for backend in ${BACKENDS[@]}
do
    if [[ ${SERVICES_START[${backend}]} ]]; then
        BACKENDS_START[${i}]=${backend}
        ((i++))
    fi
done

echo ${BACKENDS_START[@]}
