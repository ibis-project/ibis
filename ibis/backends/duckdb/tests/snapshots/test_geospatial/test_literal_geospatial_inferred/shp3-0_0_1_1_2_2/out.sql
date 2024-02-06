SELECT
  ST_ASWKB("result") AS result
FROM (
  SELECT
    ST_GEOMFROMTEXT('LINESTRING (0 0, 1 1, 2 2)') AS "result"
)