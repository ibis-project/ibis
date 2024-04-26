SELECT
  *
FROM "events" AS "t0"
WHERE
  EXISTS(
    SELECT
      1
    FROM (
      SELECT
        *
      FROM "purchases" AS "t1"
      WHERE
        "t1"."ts" > '2015-08-15'
    ) AS "t2"
    WHERE
      "t0"."user_id" = "t2"."user_id"
  )