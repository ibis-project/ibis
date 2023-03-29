SELECT
  t0.id,
  t0.personal,
  t0.family,
  t1.taken,
  t1.person,
  t1.quant,
  t1.reading,
  t2.id AS id_right,
  t2.site,
  t2.dated
FROM person AS t0
JOIN survey AS t1
  ON t0.id = t1.person
JOIN visited AS t2
  ON t2.id = t1.taken