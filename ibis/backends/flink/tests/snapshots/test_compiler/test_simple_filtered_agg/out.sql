SELECT count(DISTINCT CASE WHEN t0.`g` = 'A' THEN t0.`b` ELSE NULL END) AS `CountDistinct(b, Equals(g, 'A'))`
FROM table t0