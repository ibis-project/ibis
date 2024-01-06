SELECT
  ST_ASEWKB(ST_SETSRID("t0"."geo_point", 4326)) AS "GeoSetSRID(geo_point, 4326)"
FROM "geo" AS "t0"