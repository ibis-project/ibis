WITH t0 AS (
  SELECT t3.`uuid`, count(1) AS `count`
  FROM t t3
  GROUP BY 1
)
SELECT t1.*, t2.`last_visit`
FROM (
  SELECT t0.*
  FROM (
    SELECT t0.`uuid`, max(t0.`count`) AS `max_count`
    FROM t0
    GROUP BY 1
  ) t3
    LEFT OUTER JOIN t0
      ON (t3.`uuid` = t0.`uuid`) AND
         (t3.`max_count` = t0.`count`)
) t1
  LEFT OUTER JOIN (
    SELECT t3.`uuid`, max(t3.`ts`) AS `last_visit`
    FROM t t3
    GROUP BY 1
  ) t2
    ON t1.`uuid` = t2.`uuid`