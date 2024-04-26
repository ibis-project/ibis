SELECT
  *
FROM "foo" AS "t0"
WHERE
  "t0"."job" IN (
    SELECT
      "t1"."job"
    FROM "bar" AS "t1"
  )