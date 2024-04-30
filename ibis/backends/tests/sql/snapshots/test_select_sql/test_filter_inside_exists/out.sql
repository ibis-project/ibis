SELECT
  *
FROM "events" AS "t0"
WHERE
  EXISTS(
    SELECT
      1
    FROM "purchases" AS "t1"
    WHERE
      "t1"."ts" > '2015-08-15' AND "t0"."user_id" = "t1"."user_id"
  )