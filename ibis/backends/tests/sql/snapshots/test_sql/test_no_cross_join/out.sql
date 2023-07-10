SELECT
  t0.id AS id,
  t0.personal AS personal,
  t0.family AS family,
  t1.taken AS taken,
  t1.person AS person,
  t1.quant AS quant,
  t1.reading AS reading,
  t2.id AS id_right,
  t2.site AS site,
  t2.dated AS dated
FROM person AS t0
INNER JOIN survey AS t1
  ON t0.id = t1.person
INNER JOIN visited AS t2
  ON t2.id = t1.taken