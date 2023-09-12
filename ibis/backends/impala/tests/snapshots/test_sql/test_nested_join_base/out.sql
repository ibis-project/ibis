WITH t0 AS (
  SELECT t2.`uuid`, count(1) AS `CountStar(t)`
  FROM `t` t2
  GROUP BY 1
),
t1 AS (
  SELECT t0.`uuid`, max(t0.`CountStar(t)`) AS `max_count`
  FROM t0
  GROUP BY 1
)
SELECT t0.*
FROM t1
  LEFT OUTER JOIN t0
    ON t1.`uuid` = t0.`uuid`