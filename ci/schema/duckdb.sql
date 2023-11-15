CREATE OR REPLACE TABLE array_types (
    x BIGINT[],
    y TEXT[],
    z DOUBLE PRECISION[],
    grouper TEXT,
    scalar_column DOUBLE PRECISION,
    multi_dim BIGINT[][]
);

INSERT INTO array_types VALUES
    ([1, 2, 3], ['a', 'b', 'c'], [1.0, 2.0, 3.0], 'a', 1.0, [[], [1, 2, 3], NULL]),
    ([4, 5], ['d', 'e'], [4.0, 5.0], 'a', 2.0, []),
    ([6, NULL], ['f', NULL], [6.0, NULL], 'a', 3.0, [NULL, [], NULL]),
    ([NULL, 1, NULL], [NULL, 'a', NULL], [], 'b', 4.0, [[1], [2], [], [3, 4, 5]]),
    ([2, NULL, 3], ['b', NULL, 'c'], NULL, 'b', 5.0, NULL),
    ([4, NULL, NULL, 5], ['d', NULL, NULL, 'e'], [4.0, NULL, NULL, 5.0], 'c', 6.0, [[1, 2, 3]]);


CREATE OR REPLACE TABLE struct (
    abc STRUCT(a DOUBLE, b STRING, c BIGINT)
);

INSERT INTO struct VALUES
    ({'a': 1.0, 'b': 'banana', 'c': 2}),
    ({'a': 2.0, 'b': 'apple', 'c': 3}),
    ({'a': 3.0, 'b': 'orange', 'c': 4}),
    ({'a': NULL, 'b': 'banana', 'c': 2}),
    ({'a': 2.0, 'b': NULL, 'c': 3}),
    (NULL),
    ({'a': 3.0, 'b': 'orange', 'c': NULL});

CREATE OR REPLACE TABLE json_t (js TEXT);

INSERT INTO json_t VALUES
    ('{"a": [1,2,3,4], "b": 1}'),
    ('{"a":null,"b":2}'),
    ('{"a":"foo", "c":null}'),
    ('null'),
    ('[42,47,55]'),
    ('[]');

CREATE OR REPLACE TABLE win (g TEXT, x BIGINT NOT NULL, y BIGINT);
INSERT INTO win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);

CREATE OR REPLACE TABLE map (idx BIGINT, kv MAP(STRING, BIGINT));
INSERT INTO map VALUES
    (1, MAP(['a', 'b', 'c'], [1, 2, 3])),
    (2, MAP(['d', 'e', 'f'], [4, 5, 6]));
