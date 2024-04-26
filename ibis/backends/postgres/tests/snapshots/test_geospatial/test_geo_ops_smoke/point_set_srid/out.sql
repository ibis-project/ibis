SELECT
  ST_ASEWKB("t1"."GeoSetSRID(geo_point, 4326)") AS "GeoSetSRID(geo_point, 4326)"
FROM (
  SELECT
    ST_SETSRID("t0"."geo_point", 4326) AS "GeoSetSRID(geo_point, 4326)"
  FROM "geo" AS "t0"
) AS "t1"