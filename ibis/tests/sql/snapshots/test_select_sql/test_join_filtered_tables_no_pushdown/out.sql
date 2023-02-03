SELECT t0.`value_a`, t1.`value_b`
FROM (
  SELECT t2.*
  FROM a t2
  WHERE (t2.`year` = 2016) AND
        (t2.`month` = 2) AND
        (t2.`day` = 29)
) t0
  LEFT OUTER JOIN (
    SELECT t2.*
    FROM b t2
    WHERE (t2.`year` = 2016) AND
          (t2.`month` = 2) AND
          (t2.`day` = 29)
  ) t1
    ON (t0.`year` = t1.`year`) AND
       (t0.`month` = t1.`month`) AND
       (t0.`day` = t1.`day`)