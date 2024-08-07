SELECT
  "x",
  "y"
FROM (
  SELECT
    "t1"."x",
    "t1"."y",
    AVG("t1"."x") OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _w
  FROM (
    SELECT
      "t0"."x",
      SUM("t0"."x") OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "y"
    FROM "t" AS "t0"
  ) AS "t1"
  WHERE
    "t1"."y" <= 37
) AS _t
WHERE
  _w IS NOT NULL