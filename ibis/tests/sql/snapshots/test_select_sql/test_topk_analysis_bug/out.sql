SELECT `origin`, count(1) AS `count`
FROM (
  SELECT t2.*
  FROM airlines t2
  WHERE t2.`dest` IN ('ORD', 'JFK', 'SFO')
) t0
  LEFT SEMI JOIN (
    SELECT t2.*
    FROM (
      SELECT t3.`dest`, avg(t3.`arrdelay`) AS `Mean(arrdelay)`
      FROM airlines t3
      WHERE t3.`dest` IN ('ORD', 'JFK', 'SFO')
      GROUP BY 1
    ) t2
    ORDER BY t2.`Mean(arrdelay)` DESC
    LIMIT 10
  ) t1
    ON t0.`dest` = t1.`dest`
GROUP BY 1