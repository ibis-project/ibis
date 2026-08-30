SELECT
  *
FROM (
  SELECT
    "t1"."f",
    "t1"."c"
  FROM (
    SELECT
      *
    FROM "star1" AS "t0"
    ORDER BY
      RANDOM() ASC
  ) AS "t1"
) AS "t2"
ORDER BY
  RANDOM() ASC