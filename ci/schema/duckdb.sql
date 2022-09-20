CREATE OR REPLACE TABLE diamonds (
    carat FLOAT,
    cut TEXT,
    color TEXT,
    clarity TEXT,
    depth FLOAT,
    "table" FLOAT,
    price BIGINT,
    x FLOAT,
    y FLOAT,
    z FLOAT
);

CREATE OR REPLACE TABLE batting (
    "playerID" TEXT,
    "yearID" BIGINT,
    stint BIGINT,
    "teamID" TEXT,
    "lgID" TEXT,
    "G" BIGINT,
    "AB" BIGINT,
    "R" BIGINT,
    "H" BIGINT,
    "X2B" BIGINT,
    "X3B" BIGINT,
    "HR" BIGINT,
    "RBI" BIGINT,
    "SB" BIGINT,
    "CS" BIGINT,
    "BB" BIGINT,
    "SO" BIGINT,
    "IBB" BIGINT,
    "HBP" BIGINT,
    "SH" BIGINT,
    "SF" BIGINT,
    "GIDP" BIGINT
);

CREATE OR REPLACE TABLE awards_players (
    "playerID" TEXT,
    "awardID" TEXT,
    "yearID" BIGINT,
    "lgID" TEXT,
    tie TEXT,
    notes TEXT
);

CREATE OR REPLACE TABLE functional_alltypes (
    "index" BIGINT,
    "Unnamed: 0" BIGINT,
    id INTEGER,
    bool_col BOOLEAN,
    tinyint_col SMALLINT,
    smallint_col SMALLINT,
    int_col INTEGER,
    bigint_col BIGINT,
    float_col REAL,
    double_col DOUBLE PRECISION,
    date_string_col TEXT,
    string_col TEXT,
    timestamp_col TIMESTAMP WITHOUT TIME ZONE,
    year INTEGER,
    month INTEGER
);

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

CREATE OR REPLACE TABLE json_t (js JSON);

INSERT INTO json_t VALUES
    ('{"a": [1,2,3,4], "b": 1}'),
    ('{"a":null,"b":2}'),
    ('{"a":"foo", "c":null}'),
    ('null'),
    ('[42,47,55]'),
    ('[]');
