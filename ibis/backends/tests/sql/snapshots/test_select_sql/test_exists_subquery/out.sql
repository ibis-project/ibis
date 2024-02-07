SELECT
  "t0"."key1",
  "t0"."key2",
  "t0"."value1"
FROM "t1" AS "t0"
WHERE
  EXISTS(
    SELECT
      1
    FROM "t2" AS "t1"
    WHERE
      "t0"."key1" = "t1"."key1"
  )