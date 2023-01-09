SELECT t0.`value_a`, t1.`value_b`
FROM (
  SELECT *
  FROM a
  WHERE (`year` = 2016) AND
        (`month` = 2) AND
        (`day` = 29)
) t0
  LEFT OUTER JOIN (
    SELECT *
    FROM b
    WHERE (`year` = 2016) AND
          (`month` = 2) AND
          (`day` = 29)
  ) t1
    ON (t0.`year` = t1.`year`) AND
       (t0.`month` = t1.`month`) AND
       (t0.`day` = t1.`day`)