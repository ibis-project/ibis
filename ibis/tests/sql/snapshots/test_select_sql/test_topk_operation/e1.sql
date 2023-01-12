SELECT *
FROM tbl t0
  LEFT SEMI JOIN (
    SELECT t2.*
    FROM (
      SELECT t0.`city`, avg(t0.`v2`) AS `Mean(v2)`
      FROM tbl t0
      GROUP BY 1
    ) t2
    ORDER BY t2.`Mean(v2)` DESC
    LIMIT 10
  ) t1
    ON t0.`city` = t1.`city`