WITH t0 AS (
  SELECT t4.`uuid`, count(1) AS `CountStar(t)`
  FROM `t` t4
  GROUP BY 1
),
t1 AS (
  SELECT t0.`uuid`, max(t0.`CountStar(t)`) AS `max_count`
  FROM t0
  GROUP BY 1
),
t2 AS (
  SELECT t4.`uuid`, max(t4.`ts`) AS `last_visit`
  FROM `t` t4
  GROUP BY 1
),
t3 AS (
  SELECT t0.*
  FROM t1
    LEFT OUTER JOIN t0
      ON (t1.`uuid` = t0.`uuid`) AND
         (t1.`max_count` = t0.`CountStar(t)`)
)
SELECT t3.*, t2.`last_visit`
FROM t3
  LEFT OUTER JOIN t2
    ON t3.`uuid` = t2.`uuid`