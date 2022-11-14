SELECT `g`
FROM (
  SELECT `g`, sum(`f`) AS `total`
  FROM alltypes
  WHERE `c` > 0
  GROUP BY 1
) t0