SELECT
  ST_DWITHIN("t0"."geom", "t0"."geom", CAST(3.0 AS DOUBLE)) AS "tmp"
FROM "t" AS "t0"