SELECT t0.foo_id, t0.total, t0.value1 
FROM (SELECT t1.foo_id AS foo_id, t1.total AS total, t1.value1 AS value1 
FROM (SELECT t2.foo_id AS foo_id, t2.total AS total, t3.value1 AS value1 
FROM (SELECT t4.foo_id AS foo_id, sum(t4.f) AS total 
FROM star1 AS t4 GROUP BY t4.foo_id) AS t2 JOIN star2 AS t3 ON t2.foo_id = t3.foo_id) AS t1 
WHERE t1.total > 100) AS t0 ORDER BY t0.total DESC