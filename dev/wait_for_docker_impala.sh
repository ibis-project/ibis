#!/bin/bash -e

start=$(date +%s)

while ! sudo lxc-attach -n "$(docker inspect --format '{{.Id}}' impala)" -- impala-shell -i impala -q 'select 1'; do
  sleep 0.5
done

stop=$(date +%s)

echo "Impala took about $(($stop - $start)) seconds to become queryable"
