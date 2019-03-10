#!/bin/bash

if [ -z "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS environment variable is empty"
    exit 1
fi

if [ ! -e "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS} does not exist"
    exit 1
fi

if [ ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS} is not a file"
    exit 1
fi

if [ -z "${IBIS_TEST_DATA_DIRECTORY}" ]; then
    echo "IBIS_TEST_DATA_DIRECTORY environment variable is empty"
    exit 1
fi

if [ ! -e "${IBIS_TEST_DATA_DIRECTORY}" ]; then
    echo "IBIS_TEST_DATA_DIRECTORY=${IBIS_TEST_DATA_DIRECTORY} does not exist"
    exit 1
fi

if [ ! -d "${IBIS_TEST_DATA_DIRECTORY}" ]; then
    echo "IBIS_TEST_DATA_DIRECTORY=${IBIS_TEST_DATA_DIRECTORY} is not a directory"
    exit 1
fi

mkdir -p /tmp/ibis
cp "${GOOGLE_APPLICATION_CREDENTIALS}" /tmp/ibis/gcloud-service-key.json
