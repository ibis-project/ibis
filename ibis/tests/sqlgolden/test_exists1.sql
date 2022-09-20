SELECT t0.*
FROM foo_t t0
WHERE EXISTS (
  SELECT 1
  FROM bar_t t1
  WHERE (t0.`key1` = t1.`key1`) AND
        (t1.`key2` = 'foo')
)
