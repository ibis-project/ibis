SELECT
  *
FROM "foo" AS "t0"
WHERE
  "t0"."y" > (
    SELECT
      AVG("t2"."y") AS "Mean(y)"
    FROM (
      SELECT
        *
      FROM "foo" AS "t1"
      WHERE
        "t0"."dept_id" = "t1"."dept_id"
    ) AS "t2"
  )