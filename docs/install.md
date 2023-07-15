---
hide:
  - toc
  - navigation
  - footer
---

# Install Ibis

## Using `pip`

We suggest starting with the DuckDB backend. It's performant and fully
featured.

```sh
pip install 'ibis-framework[duckdb]'
```

If you would like to use a different backend, all of the available options are
listed below.

{% for backend in sorted(ibis.backends.base._get_backend_names()) %}
{% if backend != "spark" %}
=== "{{ backend }}"

    ```sh
    pip install 'ibis-framework[{{ backend }}]'
    ```

{% endif %}
{% endfor %}

Note that the `ibis-framework` package is _not_ the same as the `ibis` package
in PyPI. These two libraries cannot coexist in the same Python environment, as
they are both imported with the `ibis` module name.

## Using `conda` or `mamba`

<!-- prettier-ignore-start -->

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    The base `ibis-framework` package includes support for the `duckdb`
    backend. This is our recommended backend for local execution.

    ```sh
    {{ mgr }} install -c conda-forge ibis-framework
    ```

    If you would like to use a different backend, all of the available options
    are listed below.

{% for backend in sorted(ibis.backends.base._get_backend_names()) %}
{% if backend != "spark" %}
    === "{{ backend }}"

        ```sh
        {{ mgr }} install -c conda-forge ibis-{{ backend }}
        ```

{% endif %}
{% endfor %}

{% endfor %}

<!-- prettier-ignore-end -->

---

After you've successfully installed Ibis, try going through the tutorial:

<div class="install-tutorial-button" markdown>
[Go to the Tutorial](https://github.com/ibis-project/ibis-examples){ .md-button .md-button--primary }
</div>
