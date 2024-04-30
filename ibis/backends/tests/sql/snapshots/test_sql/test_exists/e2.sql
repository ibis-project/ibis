SELECT
  *
FROM "foo_t" AS "t0"
WHERE
  EXISTS(
    SELECT
      1
    FROM "bar_t" AS "t1"
    WHERE
      (
        "t0"."key1" = "t1"."key1"
      ) AND (
        "t1"."key2" = 'foo'
      )
  )