SELECT
  "t0"."job",
  "t0"."dept_id",
  "t0"."year",
  "t0"."y"
FROM "foo" AS "t0"
WHERE
  "t0"."y" > (
    SELECT
      AVG("t2"."y") AS "Mean(y)"
    FROM (
      SELECT
        "t1"."job",
        "t1"."dept_id",
        "t1"."year",
        "t1"."y"
      FROM "foo" AS "t1"
      WHERE
        "t0"."dept_id" = "t1"."dept_id"
    ) AS "t2"
  )