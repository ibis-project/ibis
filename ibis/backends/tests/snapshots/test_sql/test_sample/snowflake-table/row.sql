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
  UNIFORM(TO_DOUBLE(0.0), TO_DOUBLE(1.0), RANDOM()) <= 0.5