# Getting started with `ibis`

This is a quick tour of some basic commands and usage patterns, just to get your flippers wet.

## Install `ibis`

This quick-start guide uses the DuckDB backend. You can check out the [Install
page](../install.md) for information on how to install other backends.

```shell title="Install Ibis using pip"
$ pip install 'ibis-framework[duckdb]'
```

```shell title="Install Ibis using conda"
$ conda install ibis-framework
```

## Download a database file

Ibis can work with several file types, but at its core, it connects to existing
databases and interacts with the data there. We'll use a local database
(DuckDB) to get the hang of this.[^1]

```python title="Download an example dataset"
>>> import urllib.request
>>> urllib.request.urlretrieve(
        "https://storage.googleapis.com/ibis-tutorial-data/palmer_penguins.ddb",
        "palmer_penguins.ddb",
    )
```

## Connect using Ibis

```python title="Connect to an existing database"
>>> import ibis
>>> con = ibis.duckdb.connect("palmer_penguins.ddb")
```

We're connected! Let's take a look at what tables are available.

```python
>>> con.list_tables()
['penguins']
```

There's one table, called `penguins`. We can ask Ibis to give us an object that we can interact with.

```python
>>> penguins = con.table("penguins")
>>> penguins
AlchemyTable: penguins
  species           string
  island            string
  bill_length_mm    float64
  bill_depth_mm     float64
  flipper_length_mm int64
  body_mass_g       int64
  sex               string
  year              int64
```

Ibis is lazily evaluated, so instead of seeing the data, we see the schema of
the table, instead. To peek at the data, we can call `head` and then `to_pandas`
to get the first few rows of the table as a pandas DataFrame.

```python
>>> penguins.head().to_pandas()
  species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex  year
0  Adelie  Torgersen            39.1           18.7              181.0       3750.0    male  2007
1  Adelie  Torgersen            39.5           17.4              186.0       3800.0  female  2007
2  Adelie  Torgersen            40.3           18.0              195.0       3250.0  female  2007
3  Adelie  Torgersen             NaN            NaN                NaN          NaN    None  2007
4  Adelie  Torgersen            36.7           19.3              193.0       3450.0  female  2007
```

`to_pandas` takes the existing lazy table expression and evaluates it. If we
leave it off, you'll see the Ibis representation of the table expression that
`to_pandas` will evaluate (when you're ready!).

```python
>>> penguins.head()
r0 := AlchemyTable: penguins
  species           string
  island            string
  bill_length_mm    float64
  bill_depth_mm     float64
  flipper_length_mm int64
  body_mass_g       int64
  sex               string
  year              int64

Limit[r0, n=5]
```

!!! note "Results in pandas DataFrame"

    Ibis returns results as a pandas DataFrame using `to_pandas`, but isn't using pandas to
    perform any of the computation. The query is executed by the backend (DuckDB in
    this case). Only when `to_pandas` is called does Ibis then pull back the results
    and convert them into a DataFrame.

## Interactive mode

For the rest of this intro, we'll turn on interactive mode, which partially
executes queries to give users a preview of the results. There is a small
difference in the way the output is formatted, but otherwise this is the same
as calling `to_pandas` on the table expression with a limit of 10 result rows
returned.

```python
>>> ibis.options.interactive = True
>>> penguins.head()
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ string  │ string    │ float64        │ float64       │ int64             │ int64       │ string │ int64 │
├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┤
│ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │        3750 │ male   │  2007 │
│ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │        3800 │ female │  2007 │
│ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │        3250 │ female │  2007 │
│ Adelie  │ Torgersen │            nan │           nan │              NULL │        NULL │ NULL   │  2007 │
│ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │        3450 │ female │  2007 │
└─────────┴───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┘
```

## Common operations

Ibis has a collection of useful table methods to manipulate and query the data
in a table (or tables).

### Filter

`filter` allows you to select rows based on a condition or set of conditions.

We can filter so we only have penguins of the species Adelie:

```python
 >>> penguins.filter(penguins.species == "Adelie")
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ string  │ string    │ float64        │ float64       │ int64             │ int64       │ string │ int64 │
├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┤
│ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │        3750 │ male   │  2007 │
│ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │        3800 │ female │  2007 │
│ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │        3250 │ female │  2007 │
│ Adelie  │ Torgersen │            nan │           nan │              NULL │        NULL │ NULL   │  2007 │
│ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │        3450 │ female │  2007 │
│ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │        3650 │ male   │  2007 │
│ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │        3625 │ female │  2007 │
│ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │        4675 │ male   │  2007 │
│ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │        3475 │ NULL   │  2007 │
│ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │        4250 │ NULL   │  2007 │
│ …       │ …         │              … │             … │                 … │           … │ …      │     … │
└─────────┴───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┘
```

Or filter for Adelie penguins that reside on the island of Torgersen:

```python
>>> penguins.filter((penguins.island == "Torgersen") & (penguins.species == "Adelie"))
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ string  │ string    │ float64        │ float64       │ int64             │ int64       │ string │ int64 │
├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┤
│ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │        3750 │ male   │  2007 │
│ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │        3800 │ female │  2007 │
│ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │        3250 │ female │  2007 │
│ Adelie  │ Torgersen │            nan │           nan │              NULL │        NULL │ NULL   │  2007 │
│ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │        3450 │ female │  2007 │
│ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │        3650 │ male   │  2007 │
│ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │        3625 │ female │  2007 │
│ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │        4675 │ male   │  2007 │
│ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │        3475 │ NULL   │  2007 │
│ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │        4250 │ NULL   │  2007 │
│ …       │ …         │              … │             … │                 … │           … │ …      │     … │
└─────────┴───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┘
```

You can use any boolean comparison in a filter (although if you try to do
something like use `<` on a string, Ibis will yell at you).

### Select

Your data analysis might not require all the columns present in a given table.
`select` lets you pick out only those columns that you want to work with.

To select a column you can use the name of the column as a string:

```python
>>> penguins.select("species", "island", "year")
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ species ┃ island    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ string  │ string    │ int64 │
├─────────┼───────────┼───────┤
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ …       │ …         │     … │
└─────────┴───────────┴───────┘
```

Or you can use column objects directly (this can be convenient when paired with
tab-completion):

```python
>>> penguins.select(penguins.species, penguins.island, penguins.year)
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ species ┃ island    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ string  │ string    │ int64 │
├─────────┼───────────┼───────┤
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ …       │ …         │     … │
└─────────┴───────────┴───────┘
```

Or you can mix-and-match:

```python

>>> penguins.select("species", "island", penguins.year)
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ species ┃ island    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ string  │ string    │ int64 │
├─────────┼───────────┼───────┤
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ Adelie  │ Torgersen │  2007 │
│ …       │ …         │     … │
└─────────┴───────────┴───────┘
```

### Mutate

`mutate` lets you add new columns to your table, derived from the values of
existing columns.

```python
>>> penguins.mutate(bill_length_cm=penguins.bill_length_mm / 10)
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━┓
┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ … ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━┩
│ string  │ string    │ float64        │ float64       │ int64             │ int64       │ string │ … │
├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───┤
│ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │        3750 │ male   │ … │
│ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │        3800 │ female │ … │
│ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │        3250 │ female │ … │
│ Adelie  │ Torgersen │            nan │           nan │              NULL │        NULL │ NULL   │ … │
│ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │        3450 │ female │ … │
│ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │        3650 │ male   │ … │
│ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │        3625 │ female │ … │
│ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │        4675 │ male   │ … │
│ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │        3475 │ NULL   │ … │
│ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │        4250 │ NULL   │ … │
│ …       │ …         │              … │             … │                 … │           … │ …      │ … │
└─────────┴───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───┘
```

Notice that the table is a little too wide to display all the columns now (depending on your screen-size).
`bill_length` is now present in millimeters AND centimeters. Use a `select` to
trim down the number of columns we're looking at.

```python
>>> penguins.mutate(bill_length_cm=penguins.bill_length_mm / 10).select(
        "species",
        "island",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
        "year",
        "bill_length_cm",
    )
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ species ┃ island    ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃ bill_length_cm ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ string  │ string    │ float64       │ int64             │ int64       │ string │ int64 │ float64        │
├─────────┼───────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┼────────────────┤
│ Adelie  │ Torgersen │          18.7 │               181 │        3750 │ male   │  2007 │           3.91 │
│ Adelie  │ Torgersen │          17.4 │               186 │        3800 │ female │  2007 │           3.95 │
│ Adelie  │ Torgersen │          18.0 │               195 │        3250 │ female │  2007 │           4.03 │
│ Adelie  │ Torgersen │           nan │              NULL │        NULL │ NULL   │  2007 │            nan │
│ Adelie  │ Torgersen │          19.3 │               193 │        3450 │ female │  2007 │           3.67 │
│ Adelie  │ Torgersen │          20.6 │               190 │        3650 │ male   │  2007 │           3.93 │
│ Adelie  │ Torgersen │          17.8 │               181 │        3625 │ female │  2007 │           3.89 │
│ Adelie  │ Torgersen │          19.6 │               195 │        4675 │ male   │  2007 │           3.92 │
│ Adelie  │ Torgersen │          18.1 │               193 │        3475 │ NULL   │  2007 │           3.41 │
│ Adelie  │ Torgersen │          20.2 │               190 │        4250 │ NULL   │  2007 │           4.20 │
│ …       │ …         │             … │                 … │           … │ …      │     … │              … │
└─────────┴───────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┴────────────────┘
```

### Selectors

Typing out ALL of the column names _except_ one is a little annoying. Instead of
doing that again, we can use a `selector` to quickly select or deselect groups
of columns.

```python
>>> from ibis import selectors as s

>>> penguins.mutate(bill_length_cm=penguins.bill_length_mm / 10).select(
        ~s.matches("bill_length_mm")
        # match every column except `bill_length_mm`
    )
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ species ┃ island    ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃ bill_length_cm ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ string  │ string    │ float64       │ int64             │ int64       │ string │ int64 │ float64        │
├─────────┼───────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┼────────────────┤
│ Adelie  │ Torgersen │          18.7 │               181 │        3750 │ male   │  2007 │           3.91 │
│ Adelie  │ Torgersen │          17.4 │               186 │        3800 │ female │  2007 │           3.95 │
│ Adelie  │ Torgersen │          18.0 │               195 │        3250 │ female │  2007 │           4.03 │
│ Adelie  │ Torgersen │           nan │              NULL │        NULL │ NULL   │  2007 │            nan │
│ Adelie  │ Torgersen │          19.3 │               193 │        3450 │ female │  2007 │           3.67 │
│ Adelie  │ Torgersen │          20.6 │               190 │        3650 │ male   │  2007 │           3.93 │
│ Adelie  │ Torgersen │          17.8 │               181 │        3625 │ female │  2007 │           3.89 │
│ Adelie  │ Torgersen │          19.6 │               195 │        4675 │ male   │  2007 │           3.92 │
│ Adelie  │ Torgersen │          18.1 │               193 │        3475 │ NULL   │  2007 │           3.41 │
│ Adelie  │ Torgersen │          20.2 │               190 │        4250 │ NULL   │  2007 │           4.20 │
│ …       │ …         │             … │                 … │           … │ …      │     … │              … │
└─────────┴───────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┴────────────────┘
```

You can also use a `selector` alongside a column name.

```python
>>> penguins.select("island", s.numeric())
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┓
┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ year  ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━┩
│ string    │ float64        │ float64       │ int64             │ int64       │ int64 │
├───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼───────┤
│ Torgersen │           39.1 │          18.7 │               181 │        3750 │  2007 │
│ Torgersen │           39.5 │          17.4 │               186 │        3800 │  2007 │
│ Torgersen │           40.3 │          18.0 │               195 │        3250 │  2007 │
│ Torgersen │            nan │           nan │              NULL │        NULL │  2007 │
│ Torgersen │           36.7 │          19.3 │               193 │        3450 │  2007 │
│ Torgersen │           39.3 │          20.6 │               190 │        3650 │  2007 │
│ Torgersen │           38.9 │          17.8 │               181 │        3625 │  2007 │
│ Torgersen │           39.2 │          19.6 │               195 │        4675 │  2007 │
│ Torgersen │           34.1 │          18.1 │               193 │        3475 │  2007 │
│ Torgersen │           42.0 │          20.2 │               190 │        4250 │  2007 │
│ …         │              … │             … │                 … │           … │     … │
└───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴───────┘
```

You can read more about [`selectors`](../reference/selectors.md) in the docs!

### order_by

`order_by` arranges the values of one or more columns in ascending or descending order.

By default, `ibis` sorts in ascending order:

```python
>>> penguins.order_by(penguins.flipper_length_mm).select(
        "species", "island", "flipper_length_mm"
    )
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ species   ┃ island    ┃ flipper_length_mm ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ string    │ string    │ int64             │
├───────────┼───────────┼───────────────────┤
│ Adelie    │ Torgersen │              NULL │
│ Gentoo    │ Biscoe    │              NULL │
│ Adelie    │ Biscoe    │               172 │
│ Adelie    │ Biscoe    │               174 │
│ Adelie    │ Torgersen │               176 │
│ Adelie    │ Dream     │               178 │
│ Adelie    │ Dream     │               178 │
│ Adelie    │ Dream     │               178 │
│ Chinstrap │ Dream     │               178 │
│ Adelie    │ Dream     │               179 │
│ …         │ …         │                 … │
└───────────┴───────────┴───────────────────┘
```

You can sort in descending order using the `desc` method of a column:

```python
>>> penguins.order_by(penguins.flipper_length_mm.desc()).select(
        "species", "island", "flipper_length_mm"
    )
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ species ┃ island    ┃ flipper_length_mm ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ string  │ string    │ int64             │
├─────────┼───────────┼───────────────────┤
│ Adelie  │ Torgersen │              NULL │
│ Gentoo  │ Biscoe    │              NULL │
│ Gentoo  │ Biscoe    │               231 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ …       │ …         │                 … │
└─────────┴───────────┴───────────────────┘
```

Or you can use `ibis.desc`

```python
>>> penguins.order_by(ibis.desc("flipper_length_mm")).select(
        "species", "island", "flipper_length_mm"
    )
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ species ┃ island    ┃ flipper_length_mm ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ string  │ string    │ int64             │
├─────────┼───────────┼───────────────────┤
│ Adelie  │ Torgersen │              NULL │
│ Gentoo  │ Biscoe    │              NULL │
│ Gentoo  │ Biscoe    │               231 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ Gentoo  │ Biscoe    │               230 │
│ …       │ …         │                 … │
└─────────┴───────────┴───────────────────┘
```

### Aggregates

Ibis has several aggregate functions available to help summarize data.

`mean`, `max`, `min`, `count`, `sum` (the list goes on).

To aggregate an entire column, call the corresponding method on that column.

```python
>>> penguins.flipper_length_mm.mean()
200.91520467836258
```

You can compute multiple aggregates at once using the `aggregate` method:

```python
>>> penguins.aggregate(
        [penguins.flipper_length_mm.mean(), penguins.bill_depth_mm.max()]
    )
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Mean(flipper_length_mm) ┃ Max(bill_depth_mm) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ float64                 │ float64            │
├─────────────────────────┼────────────────────┤
│              200.915205 │               21.5 │
└─────────────────────────┴────────────────────┘
```

But `aggregate` _really_ shines when it's paired with `group_by`.

### group_by

`group_by` creates groupings of rows that have the same value for one or more columns.

But it doesn't do much on its own -- you can pair it with `aggregate` to get a result.

```python
>>> penguins.group_by("species").aggregate()
┏━━━━━━━━━━━┓
┃ species   ┃
┡━━━━━━━━━━━┩
│ string    │
├───────────┤
│ Adelie    │
│ Gentoo    │
│ Chinstrap │
└───────────┘
```

We grouped by the `species` column and handed it an "empty" aggregate command.
The result of that is a column of the unique values in the `species` column.

If we add a second column to the `group_by`, we'll get each unique pairing of the
values in those columns.

```python
>>> penguins.group_by(["species", "island"]).aggregate()
┏━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ species   ┃ island    ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━┩
│ string    │ string    │
├───────────┼───────────┤
│ Adelie    │ Torgersen │
│ Adelie    │ Biscoe    │
│ Adelie    │ Dream     │
│ Gentoo    │ Biscoe    │
│ Chinstrap │ Dream     │
└───────────┴───────────┘
```

Now, if we add an aggregation function to that, we start to really open things up.

```python

>>> penguins.group_by(["species", "island"]).aggregate(penguins.bill_length_mm.mean())
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ species   ┃ island    ┃ Mean(bill_length_mm) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ string    │ string    │ float64              │
├───────────┼───────────┼──────────────────────┤
│ Adelie    │ Torgersen │            38.950980 │
│ Adelie    │ Biscoe    │            38.975000 │
│ Adelie    │ Dream     │            38.501786 │
│ Gentoo    │ Biscoe    │            47.504878 │
│ Chinstrap │ Dream     │            48.833824 │
└───────────┴───────────┴──────────────────────┘
```

By adding that `mean` to the `aggregate`, we now have a concise way to calculate
aggregates over each of the distinct groups in the `group_by`. And we can
calculate as many aggregates as we need.

```python
>>> penguins.group_by(["species", "island"]).aggregate(
        [penguins.bill_length_mm.mean(), penguins.flipper_length_mm.max()]
    )
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ species   ┃ island    ┃ Mean(bill_length_mm) ┃ Max(flipper_length_mm) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ string    │ string    │ float64              │ int64                  │
├───────────┼───────────┼──────────────────────┼────────────────────────┤
│ Adelie    │ Torgersen │            38.950980 │                    210 │
│ Adelie    │ Biscoe    │            38.975000 │                    203 │
│ Adelie    │ Dream     │            38.501786 │                    208 │
│ Gentoo    │ Biscoe    │            47.504878 │                    231 │
│ Chinstrap │ Dream     │            48.833824 │                    212 │
└───────────┴───────────┴──────────────────────┴────────────────────────┘
```

If we need more specific groups, we can add to the `group_by`.

```python
>>> penguins.group_by(["species", "island", "sex"]).aggregate(
        [penguins.bill_length_mm.mean(), penguins.flipper_length_mm.max()]
    )
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ species ┃ island    ┃ sex    ┃ Mean(bill_length_mm) ┃ Max(flipper_length_mm) ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ string  │ string    │ string │ float64              │ int64                  │
├─────────┼───────────┼────────┼──────────────────────┼────────────────────────┤
│ Adelie  │ Torgersen │ male   │            40.586957 │                    210 │
│ Adelie  │ Torgersen │ female │            37.554167 │                    196 │
│ Adelie  │ Torgersen │ NULL   │            37.925000 │                    193 │
│ Adelie  │ Biscoe    │ female │            37.359091 │                    199 │
│ Adelie  │ Biscoe    │ male   │            40.590909 │                    203 │
│ Adelie  │ Dream     │ female │            36.911111 │                    202 │
│ Adelie  │ Dream     │ male   │            40.071429 │                    208 │
│ Adelie  │ Dream     │ NULL   │            37.500000 │                    179 │
│ Gentoo  │ Biscoe    │ female │            45.563793 │                    222 │
│ Gentoo  │ Biscoe    │ male   │            49.473770 │                    231 │
│ …       │ …         │ …      │                    … │                      … │
└─────────┴───────────┴────────┴──────────────────────┴────────────────────────┘
```

## Chaining it all together

We've already chained some Ibis calls together. We used `mutate` to create a new
column and then `select` to only view a subset of the new table. We were just
chaining `group_by` with `aggregate`.

There's nothing stopping us from putting all of these concepts together to ask
questions of the data.

How about:

- What was the largest female penguin (by body mass) on each island in the year 2008?

```python
>>> penguins.filter((penguins.sex == "female") & (penguins.year == 2008)).group_by(
        ["island"]
    ).aggregate(penguins.body_mass_g.max())
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ island    ┃ Max(body_mass_g) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ string    │ int64            │
├───────────┼──────────────────┤
│ Biscoe    │             5200 │
│ Torgersen │             3800 │
│ Dream     │             3900 │
└───────────┴──────────────────┘
```

- What about the largest male penguin (by body mass) on each island for each year of data collection?

```python
>>> penguins.filter(penguins.sex == "male").group_by(["island", "year"]).aggregate(
        penguins.body_mass_g.max().name("max_body_mass")
    ).order_by(["year", "max_body_mass"])
┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ island    ┃ year  ┃ max_body_mass ┃
┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━┩
│ string    │ int64 │ int64         │
├───────────┼───────┼───────────────┤
│ Dream     │  2007 │          4650 │
│ Torgersen │  2007 │          4675 │
│ Biscoe    │  2007 │          6300 │
│ Torgersen │  2008 │          4700 │
│ Dream     │  2008 │          4800 │
│ Biscoe    │  2008 │          6000 │
│ Torgersen │  2009 │          4300 │
│ Dream     │  2009 │          4475 │
│ Biscoe    │  2009 │          6000 │
└───────────┴───────┴───────────────┘
```

## Learn more

That's all for this quick-start guide. If you want to learn more, check out the [tutorial](https://github.com/ibis-project/ibis-examples).

[^1]:
    Horst AM, Hill AP, Gorman KB (2020).
    palmerpenguins: Palmer Archipelago (Antarctica) penguin data.
    R package version 0.1.0.
    https://allisonhorst.github.io/palmerpenguins/.
    doi: 10.5281/zenodo.3960218.
