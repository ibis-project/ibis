SELECT
  ST_ASWKB("result") AS result
FROM (
  SELECT
    ST_GEOMFROMTEXT('LINESTRING (2 2, 1 1, 0 0)') AS "result"
)