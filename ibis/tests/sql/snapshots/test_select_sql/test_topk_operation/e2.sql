SELECT *
FROM tbl t0
  LEFT SEMI JOIN (
    SELECT *
    FROM (
      SELECT `city`, count(`city`) AS `count`
      FROM tbl
      GROUP BY 1
    ) t2
    ORDER BY `count` DESC
    LIMIT 10
  ) t1
    ON t0.`city` = t1.`city`