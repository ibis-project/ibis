SELECT *
FROM tbl t0
  LEFT SEMI JOIN (
    SELECT *
    FROM (
      SELECT `city`, avg(`v2`) AS `Mean(v2)`
      FROM tbl
      GROUP BY 1
    ) t2
    ORDER BY `Mean(v2)` DESC
    LIMIT 10
  ) t1
    ON t0.`city` = t1.`city`