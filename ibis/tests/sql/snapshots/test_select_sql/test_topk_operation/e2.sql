SELECT *
FROM tbl t0
  LEFT SEMI JOIN (
    SELECT t2.*
    FROM (
      SELECT t0.`city`, count(t0.`city`) AS `count`
      FROM tbl t0
      GROUP BY 1
    ) t2
    ORDER BY t2.`count` DESC
    LIMIT 10
  ) t1
    ON t0.`city` = t1.`city`