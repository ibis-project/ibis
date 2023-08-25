SELECT t0.*
FROM table t0
WHERE ((t0.`c` > CAST(0 AS TINYINT)) OR (t0.`c` < CAST(0 AS TINYINT))) AND
      (t0.`g` IN ('A', 'B'))