SELECT
  ST_ASEWKB("t0"."<MULTIPOINT ((0 0), (1 1), (2 2))>") AS "<MULTIPOINT ((0 0), (1 1), (2 2))>"
FROM (
  SELECT
    ST_GEOMFROMTEXT('MULTIPOINT ((0 0), (1 1), (2 2))') AS "<MULTIPOINT ((0 0), (1 1), (2 2))>"
) AS "t0"