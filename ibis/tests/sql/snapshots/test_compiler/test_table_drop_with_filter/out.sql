SELECT t0.*
FROM (
  SELECT t1.`a`
  FROM (
    SELECT t3.`a`, t3.`b`, '2018-01-01T00:00:00' AS `the_date`
    FROM (
      SELECT t4.*
      FROM (
        SELECT t5.`a`, t5.`b`, t5.`c` AS `C`
        FROM t t5
      ) t4
      WHERE t4.`C` = '2018-01-01T00:00:00'
    ) t3
  ) t1
    INNER JOIN s t2
      ON t1.`b` = t2.`b`
) t0
WHERE t0.`a` < 1.0