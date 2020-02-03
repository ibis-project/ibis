#!/bin/bash

# Among the backends, the script finds those that will need to be launched
# by the `docker-compose` or those for which test datasets should be loaded.
# The choice depends on comparing each backend with elements inside the
# USER_REQUESTED_BACKENDS variable.
#
# Usage:
#  $ ./ci/backends-to-start.sh param1 param2
# * param1: string of backends
# * param2: string of backends, which then need to launched by `docker-compose`
#     (as docker's services) before working with them or for which test
#     datasets should be loaded.
#
# Example:
# current_backends=`./ci/backends-to-start.sh "omniscidb impala parquet" "omniscidb impala"`  && echo $current_backends
#
# Output:
# 'omniscidb impala'

# convert strings to arrays
BACKENDS=($(echo $1))
USER_REQUESTED_BACKENDS=($(echo $2))

# lookup table to choose backends to start
declare -A USER_REQUESTED_BACKENDS_LOOKUP
for service in ${USER_REQUESTED_BACKENDS[@]}
do
    USER_REQUESTED_BACKENDS_LOOKUP[$service]=1
done

i=0
for backend in ${BACKENDS[@]}
do
    if [[ ${SERVICES_START[${backend}]} ]]; then
        CHOSEN_BACKENDS[${i}]=${backend}
        ((i++))
    fi
done

echo ${CHOSEN_BACKENDS[@]}
