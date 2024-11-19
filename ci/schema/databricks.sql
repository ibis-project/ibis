CREATE VIEW IF NOT EXISTS diamonds AS
SELECT * FROM parquet.`/Volumes/ibis_testing/default/{user}_{python_version}/diamonds.parquet`;

CREATE VIEW IF NOT EXISTS batting AS
SELECT * FROM parquet.`/Volumes/ibis_testing/default/{user}_{python_version}/batting.parquet`;

CREATE VIEW IF NOT EXISTS awards_players AS
SELECT * FROM parquet.`/Volumes/ibis_testing/default/{user}_{python_version}/awards_players.parquet`;

CREATE VIEW IF NOT EXISTS functional_alltypes AS
SELECT * FROM parquet.`/Volumes/ibis_testing/default/{user}_{python_version}/functional_alltypes.parquet`;

CREATE VIEW IF NOT EXISTS astronauts AS
SELECT * FROM parquet.`/Volumes/ibis_testing/default/{user}_{python_version}/astronauts.parquet`;

CREATE TABLE IF NOT EXISTS `array_types` AS
    VALUES (ARRAY(CAST(1 AS BIGINT), 2, 3), ARRAY('a', 'b', 'c'), ARRAY(1.0, 2.0, 3.0), 'a', 1.0, ARRAY(ARRAY(), ARRAY(CAST(1 AS BIGINT), 2, 3), NULL)),
           (ARRAY(4, 5), ARRAY('d', 'e'), ARRAY(4.0, 5.0), 'a', 2.0, ARRAY()),
           (ARRAY(6, NULL), ARRAY('f', NULL), ARRAY(6.0, NULL), 'a', 3.0, ARRAY(NULL, ARRAY(), NULL)),
           (ARRAY(NULL, 1, NULL), ARRAY(NULL, 'a', NULL), ARRAY(), 'b', 4.0, ARRAY(ARRAY(1), ARRAY(2), ARRAY(), ARRAY(3, 4, 5))),
           (ARRAY(2, NULL, 3), ARRAY('b', NULL, 'c'), NULL, 'b', 5.0, NULL),
           (ARRAY(4, NULL, NULL, 5), ARRAY('d', NULL, NULL, 'e'), ARRAY(4.0, NULL, NULL, 5.0), 'c', 6.0, ARRAY(ARRAY(1, 2, 3)))
        AS (`x`, `y`, `z`, `grouper`, `scalar_column`, `multi_dim`);

CREATE TABLE IF NOT EXISTS `map` AS
    VALUES (CAST(1 AS BIGINT), map('a', CAST(1 AS BIGINT), 'b', 2, 'c', 3)),
           (2, map('d', 4, 'e', 5, 'f', 6)) AS (`idx`, `kv`);

CREATE TABLE IF NOT EXISTS `struct` AS
    VALUES (named_struct('a', 1.0, 'b', 'banana', 'c', CAST(2 AS BIGINT))),
           (named_struct('a', 2.0, 'b', 'apple', 'c', 3)),
           (named_struct('a', 3.0, 'b', 'orange', 'c', 4)),
           (named_struct('a', NULL, 'b', 'banana', 'c', 2)),
           (named_struct('a', 2.0, 'b', NULL, 'c', 3)),
           (NULL),
           (named_struct('a', 3.0, 'b', 'orange', 'c', NULL)) AS (`abc`);

CREATE TABLE IF NOT EXISTS `json_t` AS
    VALUES (CAST(1 AS BIGINT), parse_json('{"a": [1,2,3,4], "b": 1}')),
           (2, parse_json('{"a":null,"b":2}')),
           (3, parse_json('{"a":"foo", "c":null}')),
           (4, parse_json('null')),
           (5, parse_json('[42,47,55]')),
           (6, parse_json('[]')),
           (7, parse_json('"a"')),
           (8, parse_json('""')),
           (9, parse_json('"b"')),
           (10, NULL),
           (11, parse_json('true')),
           (12, parse_json('false')),
           (13, parse_json('42')),
           (14, parse_json('37.37')) AS (`rowid`, `js`);

CREATE TABLE IF NOT EXISTS `win` AS
VALUES
    ('a', CAST(0 AS BIGINT), CAST(3 AS BIGINT)),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1) AS (`g`, `x`, `y`);

CREATE TABLE IF NOT EXISTS `topk` AS
VALUES (CAST(1 AS BIGINT)), (1), (NULL) AS (`x`);
