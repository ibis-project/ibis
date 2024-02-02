SELECT
  "t0"."job",
  "t0"."dept_id",
  "t0"."year",
  "t0"."y"
FROM "foo" AS "t0"
WHERE
  "t0"."y" > (
    SELECT
      MAX("t1"."x") AS "Max(x)"
    FROM "bar" AS "t1"
  )