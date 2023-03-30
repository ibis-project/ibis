# Maximizing Productivity with Selectors

Before Ibis 5.0 it's been challenging to concisely express whole-table
operations with ibis. Happily this is no longer the case in ibis 5.0.

Let's jump right in!

We'll look at selectors examples using the [`palmerpenguins` data
set](https://allisonhorst.github.io/palmerpenguins/) with the [DuckDB
backend](https://ibis-project.org/backends/DuckDB/).

## Setup

```
In [8]: from ibis.interactive import *

In [11]: t = ex.penguins.fetch()

In [12]: t
Out[12]:
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

## Examples

### Normalization

Let's say you want to compute the
[z-score](https://en.wikipedia.org/wiki/Standard_score) of every numeric column
and replace the existing data with that normalized value. Here's how you'd do
that with selectors:

```
In [13]: t.mutate(s.across(s.numeric(), (_ - _.mean()) / _.std()))
Out[13]:
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year      ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│ string  │ string    │ float64        │ float64       │ float64           │ float64     │ string │ float64   │
├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────────┤
│ Adelie  │ Torgersen │      -0.883205 │      0.784300 │         -1.416272 │   -0.563317 │ male   │ -1.257484 │
│ Adelie  │ Torgersen │      -0.809939 │      0.126003 │         -1.060696 │   -0.500969 │ female │ -1.257484 │
│ Adelie  │ Torgersen │      -0.663408 │      0.429833 │         -0.420660 │   -1.186793 │ female │ -1.257484 │
│ Adelie  │ Torgersen │            nan │           nan │               nan │         nan │ NULL   │ -1.257484 │
│ Adelie  │ Torgersen │      -1.322799 │      1.088129 │         -0.562890 │   -0.937403 │ female │ -1.257484 │
│ Adelie  │ Torgersen │      -0.846572 │      1.746426 │         -0.776236 │   -0.688012 │ male   │ -1.257484 │
│ Adelie  │ Torgersen │      -0.919837 │      0.328556 │         -1.416272 │   -0.719186 │ female │ -1.257484 │
│ Adelie  │ Torgersen │      -0.864888 │      1.240044 │         -0.420660 │    0.590115 │ male   │ -1.257484 │
│ Adelie  │ Torgersen │      -1.799025 │      0.480471 │         -0.562890 │   -0.906229 │ NULL   │ -1.257484 │
│ Adelie  │ Torgersen │      -0.352029 │      1.543873 │         -0.776236 │    0.060160 │ NULL   │ -1.257484 │
│ …       │ …         │              … │             … │                 … │           … │ …      │         … │
└─────────┴───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────────┘
```

### What's Up With the `year` Column?

Whoops, looks like we included `year` in our normalization because it's an
`int64` column (and therefore numeric) but normalizing the year doesn't make
sense.

We can exclude `year` from the normalization using another selector:

```
In [14]: t.mutate(s.across(s.numeric() & ~s.c("year"), (_ - _.mean()) / _.std()))
Out[14]:
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ string  │ string    │ float64        │ float64       │ float64           │ float64     │ string │ int64 │
├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┤
│ Adelie  │ Torgersen │      -0.883205 │      0.784300 │         -1.416272 │   -0.563317 │ male   │  2007 │
│ Adelie  │ Torgersen │      -0.809939 │      0.126003 │         -1.060696 │   -0.500969 │ female │  2007 │
│ Adelie  │ Torgersen │      -0.663408 │      0.429833 │         -0.420660 │   -1.186793 │ female │  2007 │
│ Adelie  │ Torgersen │            nan │           nan │               nan │         nan │ NULL   │  2007 │
│ Adelie  │ Torgersen │      -1.322799 │      1.088129 │         -0.562890 │   -0.937403 │ female │  2007 │
│ Adelie  │ Torgersen │      -0.846572 │      1.746426 │         -0.776236 │   -0.688012 │ male   │  2007 │
│ Adelie  │ Torgersen │      -0.919837 │      0.328556 │         -1.416272 │   -0.719186 │ female │  2007 │
│ Adelie  │ Torgersen │      -0.864888 │      1.240044 │         -0.420660 │    0.590115 │ male   │  2007 │
│ Adelie  │ Torgersen │      -1.799025 │      0.480471 │         -0.562890 │   -0.906229 │ NULL   │  2007 │
│ Adelie  │ Torgersen │      -0.352029 │      1.543873 │         -0.776236 │    0.060160 │ NULL   │  2007 │
│ …       │ …         │              … │             … │                 … │           … │ …      │     … │
└─────────┴───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┘
```

`c` is short for "column" and the `~` means "negate". Combining those we get "not the year column"!

Pretty neat right?

### Composable Group By

The power of this approach comes in when you want the grouped version. Perhaps
we think some of these columns vary by species.

With selectors, all you need to do is slap a `.group_by("species")` onto `t`:

```
In [18]: t.group_by("species").mutate(
    ...:     s.across(s.numeric() & ~s.c("year"), (_ - _.mean()) / _.std())
    ...: )
Out[18]:
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ species ┃ island ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ string  │ string │ float64        │ float64       │ float64           │ float64     │ string │ int64 │
├─────────┼────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┤
│ Gentoo  │ Biscoe │      -0.455854 │     -1.816223 │         -0.954050 │   -1.142626 │ female │  2007 │
│ Gentoo  │ Biscoe │      -0.975022 │     -0.287513 │         -0.491442 │   -0.448342 │ female │  2009 │
│ Gentoo  │ Biscoe │       0.387793 │     -0.898997 │         -1.108253 │   -1.241809 │ female │  2007 │
│ Gentoo  │ Biscoe │       0.809616 │      0.222056 │          0.125368 │    1.237778 │ male   │  2007 │
│ Gentoo  │ Biscoe │       0.030865 │     -0.491341 │         -0.337240 │    0.642677 │ male   │  2007 │
│ Gentoo  │ Biscoe │      -0.326062 │     -1.510481 │         -1.108253 │   -1.043442 │ female │  2007 │
│ Gentoo  │ Biscoe │      -0.682990 │     -0.389427 │         -0.954050 │   -0.547525 │ female │  2007 │
│ Gentoo  │ Biscoe │      -0.261167 │      0.323970 │          0.279571 │    0.245943 │ male   │  2007 │
│ Gentoo  │ Biscoe │      -1.364397 │     -1.612395 │         -1.262455 │   -1.340993 │ female │  2007 │
│ Gentoo  │ Biscoe │      -0.228719 │      0.425884 │         -0.337240 │    0.146759 │ male   │  2007 │
│ …       │ …      │              … │             … │                 … │           … │ …      │     … │
└─────────┴────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┘
```

Since ibis translates this into a run-of-the-mill selection as if you had
called `select` or `mutate` without selectors, nothing special is needed for a
backend to work with these new constructs.

Let's look at some more examples.

### Min-max Normalization

Grouped min/max normalization? Easy:

```
In [22]: t.group_by("species").mutate(
    ...:     s.across(s.numeric() & ~s.c("year"), (_ - _.min()) / (_.max() - _.min()))
    ...: )
Out[22]:
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ species ┃ island ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ string  │ string │ float64        │ float64       │ float64           │ float64     │ string │ int64 │
├─────────┼────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┤
│ Gentoo  │ Biscoe │       0.278075 │      0.023810 │          0.285714 │    0.234043 │ female │  2007 │
│ Gentoo  │ Biscoe │       0.192513 │      0.380952 │          0.392857 │    0.382979 │ female │  2009 │
│ Gentoo  │ Biscoe │       0.417112 │      0.238095 │          0.250000 │    0.212766 │ female │  2007 │
│ Gentoo  │ Biscoe │       0.486631 │      0.500000 │          0.535714 │    0.744681 │ male   │  2007 │
│ Gentoo  │ Biscoe │       0.358289 │      0.333333 │          0.428571 │    0.617021 │ male   │  2007 │
│ Gentoo  │ Biscoe │       0.299465 │      0.095238 │          0.250000 │    0.255319 │ female │  2007 │
│ Gentoo  │ Biscoe │       0.240642 │      0.357143 │          0.285714 │    0.361702 │ female │  2007 │
│ Gentoo  │ Biscoe │       0.310160 │      0.523810 │          0.571429 │    0.531915 │ male   │  2007 │
│ Gentoo  │ Biscoe │       0.128342 │      0.071429 │          0.214286 │    0.191489 │ female │  2007 │
│ Gentoo  │ Biscoe │       0.315508 │      0.547619 │          0.428571 │    0.510638 │ male   │  2007 │
│ …       │ …      │              … │             … │                 … │           … │ …      │     … │
└─────────┴────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┘
```

### Casting and Munging

How about casting every column whose name ends with any of the strings `"mm"`
or `"g"` to a `float32`? No problem!

```
In [23]: t.mutate(s.across(s.endswith(("mm", "g")), _.cast("float32")))
Out[23]:
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ string  │ string    │ float32        │ float32       │ float32           │ float32     │ string │ int64 │
├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┤
│ Adelie  │ Torgersen │      39.099998 │     18.700001 │             181.0 │      3750.0 │ male   │  2007 │
│ Adelie  │ Torgersen │      39.500000 │     17.400000 │             186.0 │      3800.0 │ female │  2007 │
│ Adelie  │ Torgersen │      40.299999 │     18.000000 │             195.0 │      3250.0 │ female │  2007 │
│ Adelie  │ Torgersen │            nan │           nan │               nan │         nan │ NULL   │  2007 │
│ Adelie  │ Torgersen │      36.700001 │     19.299999 │             193.0 │      3450.0 │ female │  2007 │
│ Adelie  │ Torgersen │      39.299999 │     20.600000 │             190.0 │      3650.0 │ male   │  2007 │
│ Adelie  │ Torgersen │      38.900002 │     17.799999 │             181.0 │      3625.0 │ female │  2007 │
│ Adelie  │ Torgersen │      39.200001 │     19.600000 │             195.0 │      4675.0 │ male   │  2007 │
│ Adelie  │ Torgersen │      34.099998 │     18.100000 │             193.0 │      3475.0 │ NULL   │  2007 │
│ Adelie  │ Torgersen │      42.000000 │     20.200001 │             190.0 │      4250.0 │ NULL   │  2007 │
│ …       │ …         │              … │             … │                 … │           … │ …      │     … │
└─────────┴───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┘
```

We can make all string columns have the same case too!

```
In [35]: t.mutate(s.across(s.of_type("string"), _.lower()))
Out[35]:
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ string  │ string    │ float64        │ float64       │ int64             │ int64       │ string │ int64 │
├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────┤
│ adelie  │ torgersen │           39.1 │          18.7 │               181 │        3750 │ male   │  2007 │
│ adelie  │ torgersen │           39.5 │          17.4 │               186 │        3800 │ female │  2007 │
│ adelie  │ torgersen │           40.3 │          18.0 │               195 │        3250 │ female │  2007 │
│ adelie  │ torgersen │            nan │           nan │              NULL │        NULL │ NULL   │  2007 │
│ adelie  │ torgersen │           36.7 │          19.3 │               193 │        3450 │ female │  2007 │
│ adelie  │ torgersen │           39.3 │          20.6 │               190 │        3650 │ male   │  2007 │
│ adelie  │ torgersen │           38.9 │          17.8 │               181 │        3625 │ female │  2007 │
│ adelie  │ torgersen │           39.2 │          19.6 │               195 │        4675 │ male   │  2007 │
│ adelie  │ torgersen │           34.1 │          18.1 │               193 │        3475 │ NULL   │  2007 │
│ adelie  │ torgersen │           42.0 │          20.2 │               190 │        4250 │ NULL   │  2007 │
│ …       │ …         │              … │             … │                 … │           … │ …      │     … │
└─────────┴───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────┘
```

### Multiple Computations per Column

What if I want to compute multiple things? Heck yeah!

```
In [9]: t.group_by("sex").mutate(
   ...:     s.across(
   ...:         s.numeric() & ~s.c("year"),
   ...:         dict(centered=_ - _.mean(), zscore=(_ - _.mean()) / _.std()),
   ...:     )
   ...: ).select("sex", s.endswith(("_centered", "_zscore")))
Out[9]:
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━┓
┃ sex    ┃ bill_length_mm_centered ┃ bill_depth_mm_centered ┃ flipper_length_mm_centered ┃ body_mass_g_centered ┃ bill_length_mm_zscore ┃ bill_depth_mm_zscore ┃ flipper_length_mm_zscore ┃ … ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━┩
│ string │ float64                 │ float64                │ float64                    │ float64              │ float64               │ float64              │ float64                  │ … │
├────────┼─────────────────────────┼────────────────────────┼────────────────────────────┼──────────────────────┼───────────────────────┼──────────────────────┼──────────────────────────┼───┤
│ male   │                0.445238 │              -2.091071 │                  10.494048 │           504.315476 │              0.082960 │            -1.122210 │                 0.721346 │ … │
│ male   │                2.245238 │              -2.791071 │                   4.494048 │           954.315476 │              0.418349 │            -1.497878 │                 0.308914 │ … │
│ male   │               -6.254762 │               0.208929 │                 -18.505952 │           -95.684524 │             -1.165434 │             0.112125 │                -1.272072 │ … │
│ male   │               -5.054762 │               1.008929 │                   3.494048 │          -245.684524 │             -0.941841 │             0.541459 │                 0.240176 │ … │
│ male   │              -11.254762 │               3.208929 │                  -6.505952 │          -145.684524 │             -2.097071 │             1.722128 │                -0.447210 │ … │
│ male   │               -3.354762 │               2.808929 │                  -7.505952 │           -45.684524 │             -0.625084 │             1.507461 │                -0.515948 │ … │
│ male   │                0.145238 │               3.608929 │                 -10.505952 │          -345.684524 │              0.027062 │             1.936795 │                -0.722164 │ … │
│ male   │               -8.154762 │               0.808929 │                 -24.505952 │          -945.684524 │             -1.519456 │             0.434126 │                -1.684504 │ … │
│ male   │               -7.654762 │               0.208929 │                 -19.505952 │          -595.684524 │             -1.426292 │             0.112125 │                -1.340811 │ … │
│ male   │               -7.054762 │              -0.691071 │                 -24.505952 │          -745.684524 │             -1.314496 │            -0.370876 │                -1.684504 │ … │
│ …      │                       … │                      … │                          … │                    … │                     … │                    … │                        … │ … │
└────────┴─────────────────────────┴────────────────────────┴────────────────────────────┴──────────────────────┴───────────────────────┴──────────────────────┴──────────────────────────┴───┘
```

Don't like the naming convention?

Pass a function to make your own name!

```
In [12]: t.select(s.startswith("bill")).mutate(
    ...:     s.across(
    ...:         s.all(),
    ...:         dict(x=_ - _.mean(), y=_.max()),
    ...:         names=lambda col, fn: f"{col}_{fn}_improved",
    ...:     )
    ...: )
Out[12]:
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ bill_length_mm ┃ bill_depth_mm ┃ bill_length_mm_x_improved ┃ bill_depth_mm_x_improved ┃ bill_length_mm_y_improved ┃ bill_depth_mm_y_improved ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ float64        │ float64       │ float64                   │ float64                  │ float64                   │ float64                  │
├────────────────┼───────────────┼───────────────────────────┼──────────────────────────┼───────────────────────────┼──────────────────────────┤
│           39.1 │          18.7 │                  -4.82193 │                  1.54883 │                      59.6 │                     21.5 │
│           39.5 │          17.4 │                  -4.42193 │                  0.24883 │                      59.6 │                     21.5 │
│           40.3 │          18.0 │                  -3.62193 │                  0.84883 │                      59.6 │                     21.5 │
│            nan │           nan │                       nan │                      nan │                      59.6 │                     21.5 │
│           36.7 │          19.3 │                  -7.22193 │                  2.14883 │                      59.6 │                     21.5 │
│           39.3 │          20.6 │                  -4.62193 │                  3.44883 │                      59.6 │                     21.5 │
│           38.9 │          17.8 │                  -5.02193 │                  0.64883 │                      59.6 │                     21.5 │
│           39.2 │          19.6 │                  -4.72193 │                  2.44883 │                      59.6 │                     21.5 │
│           34.1 │          18.1 │                  -9.82193 │                  0.94883 │                      59.6 │                     21.5 │
│           42.0 │          20.2 │                  -1.92193 │                  3.04883 │                      59.6 │                     21.5 │
│              … │             … │                         … │                        … │                         … │                        … │
└────────────────┴───────────────┴───────────────────────────┴──────────────────────────┴───────────────────────────┴──────────────────────────┘
```

Don't like lambda functions? We support a format string too!

```
In [5]: t.select(s.startswith("bill")).mutate(
   ...:     s.across(
   ...:         s.all(),
   ...:         func=dict(x=_ - _.mean(), y=_.max()),
   ...:         names="{col}_{fn}_improved",
   ...:     )
   ...: ).head(2)
Out[5]:
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ bill_length_mm ┃ bill_depth_mm ┃ bill_length_mm_x_improved ┃ bill_depth_mm_x_improved ┃ bill_length_mm_y_improved ┃ bill_depth_mm_y_improved ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ float64        │ float64       │ float64                   │ float64                  │ float64                   │ float64                  │
├────────────────┼───────────────┼───────────────────────────┼──────────────────────────┼───────────────────────────┼──────────────────────────┤
│           39.1 │          18.7 │                  -4.82193 │                  1.54883 │                      59.6 │                     21.5 │
│           39.5 │          17.4 │                  -4.42193 │                  0.24883 │                      59.6 │                     21.5 │
└────────────────┴───────────────┴───────────────────────────┴──────────────────────────┴───────────────────────────┴──────────────────────────┘
```

### Working with other Ibis APIs

We've seen lots of mutate use, but selectors also work with `.agg`:

```
In [31]: t.group_by("year").agg(s.across(s.numeric() & ~s.c("year"), _.mean())).order_by("year")
Out[31]:
┏━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ year  ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ int64 │ float64        │ float64       │ float64           │ float64     │
├───────┼────────────────┼───────────────┼───────────────────┼─────────────┤
│  2007 │      43.740367 │     17.427523 │        196.880734 │ 4124.541284 │
│  2008 │      43.541228 │     16.914035 │        202.798246 │ 4266.666667 │
│  2009 │      44.452941 │     17.125210 │        202.806723 │ 4210.294118 │
└───────┴────────────────┴───────────────┴───────────────────┴─────────────┘
```

Naturally, selectors work in grouping keys too, for even more convenience:

```
In [12]: t.group_by(~s.numeric() | s.c("year")).mutate(
    ...:     s.across(s.numeric() & ~s.c("year"), dict(centered=_ - _.mean(), std=_.std()))
    ...: ).select("species", s.endswith(("_centered", "_std")))
Out[12]:
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ species ┃ bill_length_mm_centered ┃ bill_depth_mm_centered ┃ flipper_length_mm_centered ┃ body_mass_g_centered ┃ bill_length_mm_std ┃ bill_depth_mm_std ┃ flipper_length_mm_std ┃ body_mass_g_std ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ string  │ float64                 │ float64                │ float64                    │ float64              │ float64            │ float64           │ float64               │ float64         │
├─────────┼─────────────────────────┼────────────────────────┼────────────────────────────┼──────────────────────┼────────────────────┼───────────────────┼───────────────────────┼─────────────────┤
│ Adelie  │                1.187500 │               1.412500 │                       -1.0 │          -550.000000 │           2.087676 │          0.756755 │              7.764388 │      311.677489 │
│ Adelie  │               -3.812500 │               0.612500 │                       -5.0 │          -300.000000 │           2.087676 │          0.756755 │              7.764388 │      311.677489 │
│ Adelie  │               -1.812500 │               0.312500 │                       -6.0 │          -150.000000 │           2.087676 │          0.756755 │              7.764388 │      311.677489 │
│ Adelie  │                0.987500 │              -0.787500 │                       10.0 │           200.000000 │           2.087676 │          0.756755 │              7.764388 │      311.677489 │
│ Adelie  │               -0.512500 │              -0.787500 │                       -9.0 │           350.000000 │           2.087676 │          0.756755 │              7.764388 │      311.677489 │
│ Adelie  │                0.687500 │               0.012500 │                       13.0 │           200.000000 │           2.087676 │          0.756755 │              7.764388 │      311.677489 │
│ Adelie  │                0.187500 │              -0.387500 │                        1.0 │           250.000000 │           2.087676 │          0.756755 │              7.764388 │      311.677489 │
│ Adelie  │                3.087500 │              -0.387500 │                       -3.0 │             0.000000 │           2.087676 │          0.756755 │              7.764388 │      311.677489 │
│ Adelie  │                1.644444 │              -1.144444 │                       -7.0 │           -19.444444 │           2.119028 │          0.860394 │              5.408327 │      170.375403 │
│ Adelie  │                1.644444 │              -0.044444 │                        3.0 │            30.555556 │           2.119028 │          0.860394 │              5.408327 │      170.375403 │
│ …       │                       … │                      … │                          … │                    … │                  … │                 … │                     … │               … │
└─────────┴─────────────────────────┴────────────────────────┴────────────────────────────┴──────────────────────┴────────────────────┴───────────────────┴───────────────────────┴─────────────────┘
```

### Filtering Selectors

You can also express complex filters more concisely.

Let's say we only want to keep rows where all the bill size z-score related
columns' absolute values are greater than 2.

<!-- prettier-ignore-start -->
```
In [78]: t.drop("year").group_by("species").mutate(
    ...:     s.across(s.numeric(), dict(zscore=(_ - _.mean()) / _.std()))
    ...: ).filter(s.if_all(s.startswith("bill") & s.endswith("_zscore"), _.abs() > 2))
Out[78]:
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ bill_length_mm_zscore ┃ bill_depth_mm_zscore ┃ flipper_length_mm_zscore ┃ body_mass_g_zscore ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ string  │ string    │ float64        │ float64       │ int64             │ int64       │ string │ float64               │ float64              │ float64                  │ float64            │
├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┼───────────────────────┼──────────────────────┼──────────────────────────┼────────────────────┤
│ Gentoo  │ Biscoe    │           59.6 │          17.0 │               230 │        6050 │ male   │              3.924621 │             2.056508 │                 1.975799 │           1.932062 │
│ Gentoo  │ Biscoe    │           55.9 │          17.0 │               228 │        5600 │ male   │              2.724046 │             2.056508 │                 1.667394 │           1.039411 │
│ Adelie  │ Torgersen │           46.0 │          21.5 │               194 │        4200 │ male   │              2.706539 │             2.592071 │                 0.618760 │           1.088911 │
│ Adelie  │ Dream     │           32.1 │          15.5 │               188 │        3050 │ female │             -2.512345 │            -2.339505 │                -0.298747 │          -1.418906 │
└─────────┴───────────┴────────────────┴───────────────┴───────────────────┴─────────────┴────────┴───────────────────────┴──────────────────────┴──────────────────────────┴────────────────────┘```
```
<!-- prettier-ignore-end -->

### Bonus: Generated SQL

The SQL for that last expression is pretty gnarly:

```
In [79]: ibis.show_sql(
    ...:     t.drop("year")
    ...:     .group_by("species")
    ...:     .mutate(s.across(s.numeric(), dict(zscore=(_ - _.mean()) / _.std())))
    ...:     .filter(s.if_all(s.startswith("bill") & s.endswith("_zscore"), _.abs() > 2))
    ...: )
```

```sql
WITH t0 AS (
  SELECT
    t2.species AS species,
    t2.island AS island,
    t2.bill_length_mm AS bill_length_mm,
    t2.bill_depth_mm AS bill_depth_mm,
    t2.flipper_length_mm AS flipper_length_mm,
    t2.body_mass_g AS body_mass_g,
    t2.sex AS sex
  FROM ibis_read_csv_3 AS t2
), t1 AS (
  SELECT
    t0.species AS species,
    t0.island AS island,
    t0.bill_length_mm AS bill_length_mm,
    t0.bill_depth_mm AS bill_depth_mm,
    t0.flipper_length_mm AS flipper_length_mm,
    t0.body_mass_g AS body_mass_g,
    t0.sex AS sex,
    (
      t0.bill_length_mm - AVG(t0.bill_length_mm) OVER (PARTITION BY t0.species ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    ) / STDDEV_SAMP(t0.bill_length_mm) OVER (PARTITION BY t0.species ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS bill_length_mm_zscore,
    (
      t0.bill_depth_mm - AVG(t0.bill_depth_mm) OVER (PARTITION BY t0.species ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    ) / STDDEV_SAMP(t0.bill_depth_mm) OVER (PARTITION BY t0.species ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS bill_depth_mm_zscore,
    (
      t0.flipper_length_mm - AVG(t0.flipper_length_mm) OVER (PARTITION BY t0.species ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    ) / STDDEV_SAMP(t0.flipper_length_mm) OVER (PARTITION BY t0.species ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS flipper_length_mm_zscore,
    (
      t0.body_mass_g - AVG(t0.body_mass_g) OVER (PARTITION BY t0.species ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    ) / STDDEV_SAMP(t0.body_mass_g) OVER (PARTITION BY t0.species ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS body_mass_g_zscore
  FROM t0
)
SELECT
  t1.species,
  t1.island,
  t1.bill_length_mm,
  t1.bill_depth_mm,
  t1.flipper_length_mm,
  t1.body_mass_g,
  t1.sex,
  t1.bill_length_mm_zscore,
  t1.bill_depth_mm_zscore,
  t1.flipper_length_mm_zscore,
  t1.body_mass_g_zscore
FROM t1
WHERE
  ABS(t1.bill_length_mm_zscore) > CAST(2 AS SMALLINT)
  AND ABS(t1.bill_depth_mm_zscore) > CAST(2 AS SMALLINT)
```

Good thing you didn't have to write that by hand!

## Summary

This blog post illustrates the ability to apply computations to many columns at
once and the power of ibis as a composable, expressive library for analytics.

- [Get involved!](https://ibis-project.org/community/contribute/)
- [Report issues!](https://github.com/ibis-project/ibis/issues/new/choose)
