WITH t0 AS (
  SELECT `userid`, `movieid`, `rating`,
         CAST(`timestamp` AS timestamp) AS `datetime`
  FROM ratings
),
t1 AS (
  SELECT t0.*, t5.`title`
  FROM t0
    INNER JOIN movies t5
      ON t0.`movieid` = t5.`movieid`
)
SELECT t2.*
FROM (
  SELECT t1.*
  FROM t1
  WHERE (t1.`userid` = 118205) AND
        (extract(t1.`datetime`, 'year') > 2001)
) t2
WHERE t2.`movieid` IN (
  SELECT `movieid`
  FROM (
    SELECT `movieid`
    FROM (
      SELECT t1.*
      FROM t1
      WHERE (t1.`userid` = 118205) AND
            (extract(t1.`datetime`, 'year') > 2001) AND
            (t1.`userid` = 118205) AND
            (extract(t1.`datetime`, 'year') < 2009)
    ) t5
  ) t4
)