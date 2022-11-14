SELECT t0.*
FROM (
  SELECT t2.`a`
  FROM (
    SELECT `a`, `b`, '2018-01-01T00:00:00' AS `the_date`
    FROM (
      SELECT *
      FROM (
        SELECT `a`, `b`, `c` AS `C`
        FROM t
      ) t5
      WHERE `C` = '2018-01-01T00:00:00'
    ) t4
  ) t2
    INNER JOIN s t1
      ON t2.`b` = t1.`b`
) t0
WHERE t0.`a` < 1.0