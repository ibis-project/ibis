SELECT t0.foo_id, t0.total, t0.value1 
FROM (SELECT t1.foo_id AS foo_id, t1.total AS total, t1.value1 AS value1 
FROM (SELECT t4.foo_id AS foo_id, t4.total AS total, t2.value1 AS value1 
FROM (SELECT t3.foo_id AS foo_id, sum(t3.f) AS total 
FROM star1 AS t3 GROUP BY t3.foo_id) AS t4 JOIN star2 AS t2 ON t4.foo_id = t2.foo_id) AS t1 
WHERE t1.total > 100) AS t0 ORDER BY t0.total DESC