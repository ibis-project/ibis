SELECT
  *
FROM (
  SELECT
    *
  FROM "test" AS "t0"
  WHERE
    "t0"."x" > 10
) AS "t1"
WHERE
  randCanonical() <= 0.5