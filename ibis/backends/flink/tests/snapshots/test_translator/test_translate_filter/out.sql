SELECT t0.*
FROM table t0
WHERE ((t0.`c` > 0) OR (t0.`c` < 0)) AND
      (t0.`g` IN ('A', 'B'))