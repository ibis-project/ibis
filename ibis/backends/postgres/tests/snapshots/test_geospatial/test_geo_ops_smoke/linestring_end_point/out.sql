SELECT
  ST_ASEWKB(ST_ENDPOINT("t0"."geo_linestring")) AS "GeoEndPoint(geo_linestring)"
FROM "geo" AS "t0"