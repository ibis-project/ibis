#!/bin/bash

mkdir -p /tmp/ibis
cp "${GOOGLE_APPLICATION_CREDENTIALS}" /tmp/ibis/gcloud-service-key.json
cp -rf "${HOME}/data/ibis-testing-data" /tmp/ibis
tar -I pigz -v -cf /tmp/ibis/ibis-testing-data.tar.gz "${HOME}/data/ibis-testing-data"
