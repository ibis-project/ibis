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
) AS "t4"
WHERE
  "t4"."on_right" = (
    SELECT
      MAX("t5"."on") AS "Max(on)"
    FROM (
      SELECT
        *
      FROM "right" AS "t1"
      WHERE
        "t1"."by" = "t4"."by" AND "t1"."on" <= "t4"."on"
    ) AS "t5"
  )