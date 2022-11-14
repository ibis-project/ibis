SELECT t0.foo_id, t0.total, t1.value1 
FROM (SELECT t2.foo_id AS foo_id, sum(t2.f) AS total 
FROM star1 AS t2 GROUP BY t2.foo_id) AS t0 JOIN star2 AS t1 ON t0.foo_id = t1.foo_id