DROP TABLE IF EXISTS diamonds CASCADE;

CREATE TABLE diamonds (
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

DROP TABLE IF EXISTS batting CASCADE;

CREATE TABLE batting (
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

DROP TABLE IF EXISTS awards_players CASCADE;

CREATE TABLE awards_players (
    "playerID" TEXT,
    "awardID" TEXT,
    "yearID" BIGINT,
    "lgID" TEXT,
    tie TEXT,
    notes TEXT
);

DROP TABLE IF EXISTS functional_alltypes CASCADE;

CREATE TABLE functional_alltypes (
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

CREATE INDEX "ix_functional_alltypes_index" ON functional_alltypes ("index");

DROP TABLE IF EXISTS tzone CASCADE;

CREATE TABLE tzone (
    ts TIMESTAMP WITH TIME ZONE,
    key TEXT,
    value DOUBLE PRECISION
);

INSERT INTO tzone
    SELECT
	CAST('2017-05-28 11:01:31.000400' AS TIMESTAMP WITH TIME ZONE) +
	    t * INTERVAL '1 day 1 microsecond' AS ts,
	CHR(97 + t) AS key,
	t + t / 10.0 AS value
    FROM generate_series(0, 9) AS t;

DROP TABLE IF EXISTS array_types CASCADE;

CREATE TABLE IF NOT EXISTS array_types (
    x BIGINT[],
    y TEXT[],
    z DOUBLE PRECISION[],
    grouper TEXT,
    scalar_column DOUBLE PRECISION
);

INSERT INTO array_types VALUES
    (ARRAY[1, 2, 3], ARRAY['a', 'b', 'c'], ARRAY[1.0, 2.0, 3.0], 'a', 1.0),
    (ARRAY[4, 5], ARRAY['d', 'e'], ARRAY[4.0, 5.0], 'a', 2.0),
    (ARRAY[6, NULL], ARRAY['f', NULL], ARRAY[6.0, NULL], 'a', 3.0),
    (ARRAY[NULL, 1, NULL], ARRAY[NULL, 'a', NULL], ARRAY[]::DOUBLE PRECISION[], 'b', 4.0),
    (ARRAY[2, NULL, 3], ARRAY['b', NULL, 'c'], NULL, 'b', 5.0),
    (ARRAY[4, NULL, NULL, 5], ARRAY['d', NULL, NULL, 'e'], ARRAY[4.0, NULL, NULL, 5.0], 'c', 6.0);
