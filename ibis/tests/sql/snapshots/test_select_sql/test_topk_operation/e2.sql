WITH t0 AS (
  SELECT t2.`city`, count(t2.`city`) AS `Count(city)`
  FROM tbl t2
  GROUP BY 1
),
t1 AS (
  SELECT t0.*
  FROM t0
  ORDER BY t0.`Count(city)` DESC
  LIMIT 10
)
SELECT *
FROM tbl t2
  LEFT SEMI JOIN t1
    ON t2.`city` = t1.`city`