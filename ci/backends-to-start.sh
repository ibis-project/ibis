#!/bin/bash

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
