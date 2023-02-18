WITH t0 AS (
  SELECT t1.`city`, count(t1.`city`) AS `count`
  FROM tbl t1
  GROUP BY 1
)
SELECT *
FROM tbl t1
  LEFT SEMI JOIN (
    SELECT t0.*
    FROM t0
    ORDER BY t0.`count` DESC
    LIMIT 10
  ) t2
    ON t1.`city` = t2.`city`