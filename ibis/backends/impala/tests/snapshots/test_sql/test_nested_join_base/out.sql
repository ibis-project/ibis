WITH t0 AS (
  SELECT `uuid`, count(1) AS `count`
  FROM t
  GROUP BY 1
)
SELECT t0.*
FROM (
  SELECT `uuid`, max(`count`) AS `max_count`
  FROM t0
  GROUP BY 1
) t1
  LEFT OUTER JOIN t0
    ON t1.`uuid` = t0.`uuid`