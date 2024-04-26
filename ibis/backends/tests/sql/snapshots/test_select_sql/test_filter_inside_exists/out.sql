SELECT
  "t0"."session_id",
  "t0"."user_id",
  "t0"."event_type",
  "t0"."ts"
FROM "events" AS "t0"
WHERE
  EXISTS(
    SELECT
      1
    FROM (
      SELECT
        "t1"."item_id",
        "t1"."user_id",
        "t1"."price",
        "t1"."ts"
      FROM "purchases" AS "t1"
      WHERE
        "t1"."ts" > '2015-08-15'
    ) AS "t2"
    WHERE
      "t0"."user_id" = "t2"."user_id"
  )