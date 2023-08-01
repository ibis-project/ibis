{% set memtable_badges = {"native": "56ae57", "fallback": "goldenrod", "none": "ff6961"} %}

## `ibis.memtable` Support ![memtable](https://img.shields.io/badge/{{ memtable_impl }}-{{ memtable_badges[memtable_impl] }})

{% if memtable_impl == "not_implemented" %}

The {{ backend_name }} backend does not currently support in-memory tables.

Please [file an issue](https://github.com/ibis-project/ibis/issues/new/choose)
if you'd like the {{ backend_name }} backend to support in-memory tables.

{% else %}

The {{ backend_name }} backend supports `memtable`s {% if memtable_impl == "fallback" %} by constructing a string with the contents of the in-memory object. **This will be very inefficient for medium to large in-memory tables**. Please [file an issue](https://github.com/ibis-project/ibis/issues/new/choose) if you observe performance issues when using in-memory tables. {% elif memtable_impl == "native" %} by natively executing queries against the underlying storage (e.g., pyarrow Tables or pandas DataFrames).

{% endif %}

{% endif %}
