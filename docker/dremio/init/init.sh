#!/bin/sh

set -e
set -f
set -u
set -x

DATASETS="awards_players.csv batting.csv diamonds.csv functional_alltypes.parquet"
TOKEN=$(curl \
    --fail-with-body \
    -X POST \
    -H "Content-Type: application/json" \
    --data '{"userName":"dremio","password":"dremio123"}' \
    "http://dremio:9047/apiv2/login" | jq -r '.token')

response=$(curl \
    --output response.txt \
    -w "%{http_code}" \
    -X PUT \
    -H "Authorization: ${TOKEN}" \
    -H "Content-Type: application/json" \
    --data '{"name": "ibis","config":{"path":"/data/ibis/","defaultCtasFormat":"ICEBERG","propertyList":[]},"type":"NAS"}' \
    'http://dremio:9047/apiv2/source/ibis/')
if [ "$response" != 409 ] && [ "$response" != 200 ]; then
    cat response.txt
    exit 1
fi

for dataset in $DATASETS; do
    filetype=${dataset#*.}

    case $filetype in
        csv)
            curl \
                --fail-with-body \
                -X PUT \
                -H "Authorization: ${TOKEN}" \
                -H "Content-Type: application/json" \
                --data '{"fieldDelimiter":",","quote":"\"","comment":"#","lineDelimiter":"\n","escape":"\"","extractHeader":true,"trimHeader":false,"skipFirstLine":false,"type":"Text"}' \
                "http://dremio:9047/apiv2/source/ibis/file_format/${dataset}"
            ;;
        parquet)
            curl \
                --fail-with-body \
                -X PUT \
                -H "Authorization: ${TOKEN}" \
                -H "Content-Type: application/json" \
                --data '{"type": "Parquet"}' \
                "http://dremio:9047/apiv2/source/ibis/file_format/${dataset}"
            ;;
        *)
            exit 1
            ;;
    esac
done

response=$(curl \
               --output response.txt \
               -w "%{http_code}" \
               -X POST \
               -H "Authorization: ${TOKEN}" \
               -H "Content-Type: application/json" \
               --data "{\"entityType\":\"dataset\",\"type\":\"VIRTUAL_DATASET\",\"path\":[\"@dremio\", \"awards_players\"],\"sql\":\"SELECT playerID, awardID, CAST(yearID AS INT) AS yearID, lgID, tie, notes FROM \\\"ibis\\\".\\\"awards_players.csv\\\"\"}" \
               'http://dremio:9047/api/v3/catalog/')
if [ "$response" != 409 ] && [ "$response" != 200 ]; then
    cat response.txt
    exit 1
fi

response=$(curl \
               --output response.txt \
               -w "%{http_code}" \
               -X POST \
               -H "Authorization: ${TOKEN}" \
               -H "Content-Type: application/json" \
               --data "{\"entityType\":\"dataset\",\"type\":\"VIRTUAL_DATASET\",\"path\":[\"@dremio\", \"batting\"],\"sql\":\"SELECT playerID, CAST(yearID AS INT) as yearID, CAST(stint AS INT) AS stint, teamID, lgID, CAST(NULLIF(G, '') AS INT) AS G, CAST(NULLIF(AB, '') AS INT) AS AB, CAST(NULLIF(R, '') AS INT) AS R, CAST(NULLIF(H, '') AS INT) AS H, CAST(NULLIF(X2B, '') AS INT) AS X2B, CAST(NULLIF(X3B, '') AS INT) AS X3B, CAST(NULLIF(HR, '') AS INT) AS HR, CAST(NULLIF(RBI, '') AS INT) AS RBI, CAST(NULLIF(SB, '') AS INT) AS SB, CAST(NULLIF(CS, '') AS INT) AS CS, CAST(NULLIF(BB, '') AS INT) AS BB, CAST(NULLIF(SO, '') AS INT) AS SO, IBB, HBP, SH, SF, GIDP FROM \\\"ibis\\\".\\\"batting.csv\\\"\"}" \
               'http://dremio:9047/api/v3/catalog/')
if [ "$response" != 409 ] && [ "$response" != 200 ]; then
    cat response.txt
    exit 1
fi

response=$(curl \
               --output response.txt \
               -w "%{http_code}" \
               -X POST \
               -H "Authorization: ${TOKEN}" \
               -H "Content-Type: application/json" \
               --data "{\"entityType\":\"dataset\",\"type\":\"VIRTUAL_DATASET\",\"path\":[\"@dremio\", \"diamonds\"],\"sql\":\"SELECT CAST(carat AS DECIMAL) as carat, cut, color, clarity, CAST(depth AS DECIMAL) as depth, CAST(\\\"table\\\" AS DECIMAL) as \\\"table\\\", CAST(price AS INT) as price, CAST(x AS DECIMAL) as x, CAST(y AS DECIMAL) as y, CAST(z AS DECIMAL) as z FROM \\\"ibis\\\".\\\"diamonds.csv\\\"\"}" \
               'http://dremio:9047/api/v3/catalog/')
if [ "$response" != 409 ] && [ "$response" != 200 ]; then
    cat response.txt
    exit 1
fi

response=$(curl \
               --output response.txt \
               -w "%{http_code}" \
               -X POST \
               -H "Authorization: ${TOKEN}" \
               -H "Content-Type: application/json" \
               --data "{\"entityType\":\"dataset\",\"type\":\"VIRTUAL_DATASET\",\"path\":[\"@dremio\", \"functional_alltypes\"],\"sql\":\"SELECT * FROM \\\"ibis\\\".\\\"functional_alltypes.parquet\\\"\"}" \
               'http://dremio:9047/api/v3/catalog/')
if [ "$response" != 409 ] && [ "$response" != 200 ]; then
    cat response.txt
    exit 1
fi
