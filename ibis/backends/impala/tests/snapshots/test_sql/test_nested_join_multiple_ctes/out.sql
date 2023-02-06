WITH t0 AS (
  SELECT t2.`userid`, t2.`movieid`, t2.`rating`,
         CAST(t2.`timestamp` AS timestamp) AS `datetime`
  FROM ratings t2
),
t1 AS (
  SELECT t0.*, t3.`title`
  FROM t0
    INNER JOIN movies t3
      ON t0.`movieid` = t3.`movieid`
)
SELECT t1.*
FROM t1
WHERE (t1.`userid` = 118205) AND
      (extract(t1.`datetime`, 'year') > 2001) AND
      (t1.`movieid` IN (
  SELECT t2.`movieid`
  FROM (
    SELECT t3.`movieid`
    FROM (
      SELECT t1.*
      FROM t1
      WHERE (t1.`userid` = 118205) AND
            (extract(t1.`datetime`, 'year') > 2001) AND
            (t1.`userid` = 118205) AND
            (extract(t1.`datetime`, 'year') < 2009)
    ) t3
  ) t2
))