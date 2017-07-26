#!/bin/bash

function docker_run() {
    local container_id="$(docker inspect --format '{{.Id}}' impala)"
    echo "${container_id}"
    sudo lxc-attach \
	-n "${container_id}" \
	-f "/var/lib/docker/containers/${container_id}/config.lxc" -- $*
}

docker_run $*
