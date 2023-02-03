WITH t0 AS (
  SELECT t3.`userid`, t3.`movieid`, t3.`rating`,
         CAST(t3.`timestamp` AS timestamp) AS `datetime`
  FROM ratings t3
),
t1 AS (
  SELECT t0.*, t4.`title`
  FROM t0
    INNER JOIN movies t4
      ON t0.`movieid` = t4.`movieid`
)
SELECT t2.*
FROM (
  SELECT t1.*
  FROM t1
  WHERE (t1.`userid` = 118205) AND
        (extract(t1.`datetime`, 'year') > 2001)
) t2
WHERE t2.`movieid` IN (
  SELECT t3.`movieid`
  FROM (
    SELECT t4.`movieid`
    FROM (
      SELECT t1.*
      FROM t1
      WHERE (t1.`userid` = 118205) AND
            (extract(t1.`datetime`, 'year') > 2001) AND
            (t1.`userid` = 118205) AND
            (extract(t1.`datetime`, 'year') < 2009)
    ) t4
  ) t3
)