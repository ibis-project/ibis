WITH t0 AS (
  SELECT t2.`city`, avg(t2.`v2`) AS `Mean(v2)`
  FROM tbl t2
  GROUP BY 1
),
t1 AS (
  SELECT t0.*
  FROM t0
  ORDER BY t0.`Mean(v2)` DESC
  LIMIT 10
)
SELECT *
FROM tbl t2
  LEFT SEMI JOIN t1
    ON t2.`city` = t1.`city`