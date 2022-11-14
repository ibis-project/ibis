SELECT t0.job, t0.dept_id, t0.year, t0.y 
FROM foo AS t0 
WHERE t0.job IN (SELECT bar.job 
FROM bar)