WITH t0 AS 
(SELECT t2.g AS g, sum(t2.f) AS metric 
FROM alltypes AS t2 GROUP BY t2.g)
 SELECT t0.g, t0.metric 
FROM t0 JOIN t0 AS t1 ON t0.g = t1.g