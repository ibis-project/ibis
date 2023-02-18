WITH t0 AS (
  SELECT t1.`city`, avg(t1.`v2`) AS `Mean(v2)`
  FROM tbl t1
  GROUP BY 1
)
SELECT *
FROM tbl t1
  LEFT SEMI JOIN (
    SELECT t0.*
    FROM t0
    ORDER BY t0.`Mean(v2)` DESC
    LIMIT 10
  ) t2
    ON t1.`city` = t2.`city`