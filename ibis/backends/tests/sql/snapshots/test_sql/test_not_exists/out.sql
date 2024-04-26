SELECT
  *
FROM "foo_t" AS "t0"
WHERE
  NOT (
    EXISTS(
      SELECT
        1
      FROM "bar_t" AS "t1"
      WHERE
        "t0"."key1" = "t1"."key1"
    )
  )