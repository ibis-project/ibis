SELECT
  *
FROM "star1" AS "t0"
WHERE
  "t0"."f" > LN(
    (
      SELECT
        AVG("t1"."f") AS "Mean(f)"
      FROM (
        SELECT
          *
        FROM "star1" AS "t0"
        WHERE
          "t0"."foo_id" = 'foo'
      ) AS "t1"
    )
  )