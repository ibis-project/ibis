WITH t0 AS (
  SELECT t2.*
  FROM t2
  WHERE (t2.`userid` = 118205) AND
        (extract(t2.`datetime`, 'year') > 2001) AND
        (t2.`userid` = 118205) AND
        (extract(t2.`datetime`, 'year') < 2009)
),
t1 AS (
  SELECT t3.`userid`, t3.`movieid`, t3.`rating`,
         CAST(t3.`timestamp` AS timestamp) AS `datetime`
  FROM `ratings` t3
),
t2 AS (
  SELECT t1.*, t3.`title`
  FROM t1
    INNER JOIN `movies` t3
      ON t1.`movieid` = t3.`movieid`
)
SELECT t2.*
FROM t2
WHERE (t2.`userid` = 118205) AND
      (extract(t2.`datetime`, 'year') > 2001) AND
      (t2.`movieid` IN (
  SELECT t3.`movieid`
  FROM (
    SELECT t0.`movieid`
    FROM t0
  ) t3
))