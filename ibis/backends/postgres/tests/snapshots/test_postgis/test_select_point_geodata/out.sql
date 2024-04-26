SELECT
  ST_ASEWKB("t1"."geo_point") AS "geo_point"
FROM (
  SELECT
    "t0"."geo_point"
  FROM "geo" AS "t0"
) AS "t1"