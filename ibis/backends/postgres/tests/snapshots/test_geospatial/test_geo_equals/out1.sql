SELECT
  "t1"."id",
  ST_ASEWKB("t1"."geo_point") AS "geo_point",
  ST_ASEWKB("t1"."geo_linestring") AS "geo_linestring",
  ST_ASEWKB("t1"."geo_polygon") AS "geo_polygon",
  ST_ASEWKB("t1"."geo_multipolygon") AS "geo_multipolygon",
  "t1"."Location_Latitude",
  "t1"."Latitude"
FROM (
  SELECT
    "t0"."id",
    "t0"."geo_point",
    "t0"."geo_linestring",
    "t0"."geo_polygon",
    "t0"."geo_multipolygon",
    ST_Y("t0"."geo_point") AS "Location_Latitude",
    ST_Y("t0"."geo_point") AS "Latitude"
  FROM "geo" AS "t0"
) AS "t1"