SELECT `origin`, count(1) AS `count`
FROM (
  SELECT t1.*
  FROM (
    SELECT *
    FROM airlines
    WHERE `dest` IN ('ORD', 'JFK', 'SFO')
  ) t1
    LEFT SEMI JOIN (
      SELECT *
      FROM (
        SELECT `dest`, avg(`arrdelay`) AS `mean`
        FROM airlines
        GROUP BY 1
      ) t3
      ORDER BY `mean` DESC
      LIMIT 10
    ) t2
      ON `dest` = t2.`dest`
) t0
GROUP BY 1