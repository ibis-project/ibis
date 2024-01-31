SELECT
  "t0"."c",
  "t0"."f",
  "t0"."foo_id",
  "t0"."bar_id"
FROM "star1" AS "t0"
WHERE
  "t0"."f" > (
    LN(
      (
        SELECT
          AVG("t1"."f") AS "Mean(f)"
        FROM (
          SELECT
            "t0"."c",
            "t0"."f",
            "t0"."foo_id",
            "t0"."bar_id"
          FROM "star1" AS "t0"
          WHERE
            "t0"."foo_id" = 'foo'
        ) AS "t1"
      )
    ) + CAST(1 AS TINYINT)
  )