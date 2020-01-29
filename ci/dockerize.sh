#!/bin/bash

DOCKERIZE_CALL="dockerize"

add_wait() {
    wait_string=$1
    DOCKERIZE_CALL="${DOCKERIZE_CALL} ${wait_string}"
}

for service in $@; do
    case "${service}" in
    omniscidb)
        add_wait "-wait tcp://omniscidb:6274"
        ;;
    mysql)
        add_wait "-wait tcp://mysql:3306"
        ;;
    postgres)
        add_wait "-wait tcp://postgres:5432"
        ;;
    impala)
        add_wait "-wait tcp://impala:21050"
        add_wait "-wait tcp://impala:50070"
        ;;
    kudu-master)
        add_wait "-wait tcp://kudu-master:7051"
        add_wait "-wait tcp://kudu-master:8051"
        ;;
    kudu-tserver)
        add_wait "-wait tcp://kudu-tserver:7050"
        add_wait "-wait tcp://kudu-tserver:8050"
        ;;
    clickhouse)
        add_wait "-wait tcp://clickhouse:9000"
        ;;
    esac
done

DOCKERIZE_CALL="${DOCKERIZE_CALL} -wait-retry-interval 5s -timeout 10m"

echo ${DOCKERIZE_CALL}
