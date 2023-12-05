WITH t0 AS (
  SELECT t2.*
  FROM b t2
  WHERE (t2.`year` = 2016) AND
        (t2.`month` = 2) AND
        (t2.`day` = 29)
),
t1 AS (
  SELECT t2.*
  FROM a t2
  WHERE (t2.`year` = 2016) AND
        (t2.`month` = 2) AND
        (t2.`day` = 29)
)
SELECT t1.`value_a`, t0.`value_b`
FROM t1
  LEFT OUTER JOIN t0
    ON (t1.`year` = t0.`year`) AND
       (t1.`month` = t0.`month`) AND
       (t1.`day` = t0.`day`)