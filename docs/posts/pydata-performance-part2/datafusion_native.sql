SELECT
  month,
  ext,
  COUNT(DISTINCT project_name) AS project_count
FROM (
  SELECT
    project_name,
    DATE_TRUNC('month', uploaded_on) AS month,
    NULLIF(
      REPLACE(
        REPLACE(
          REPLACE(
            REGEXP_REPLACE(
              REGEXP_REPLACE(
                REGEXP_MATCH(path, CONCAT('(', '\.([a-z0-9]+)$', ')'))[2],
                'cxx|cpp|cc|c|hpp|h',
                'C/C++',
                'g'
              ),
              '^f.*$',
              'Fortran',
              'g'
            ),
            'rs',
            'Rust'
          ),
          'go',
          'Go'
        ),
        'asm',
        'Assembly'
      ),
      ''
    ) AS ext
  FROM pypi
  WHERE COALESCE(
      ARRAY_LENGTH(
        REGEXP_MATCH(path, '\.(asm|c|cc|cpp|cxx|h|hpp|rs|[Ff][0-9]{0,2}(?:or)?|go)$')
      ) > 0,
      FALSE
    )
    AND NOT COALESCE(ARRAY_LENGTH(REGEXP_MATCH(path, '(^|/)test(|s|ing)')) > 0, FALSE)
    AND NOT STRPOS(path, '/site-packages/') > 0
)
WHERE ext IS NOT NULL
GROUP BY month, ext
ORDER BY month DESC, project_count DESC
