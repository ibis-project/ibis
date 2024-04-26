SELECT
  *
FROM (
  SELECT
    "t2"."on",
    "t2"."by",
    "t3"."on" AS "on_right",
    "t3"."by" AS "by_right",
    "t3"."val"
  FROM "left" AS "t2"
  LEFT OUTER JOIN "right" AS "t3"
    ON "t2"."by" = "t3"."by"
) AS "t5"
WHERE
  "t5"."on_right" = (
    SELECT
      MAX("t4"."on") AS "Max(on)"
    FROM (
      SELECT
        *
      FROM "right" AS "t1"
      WHERE
        "t1"."by" = "t0"."by" AND "t1"."on" <= "t0"."on"
    ) AS "t4"
  )