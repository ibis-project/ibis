---
hide:
  - toc
  - navigation
  - footer
---

# Install Ibis

=== "pip"

    ```sh
    pip install 'ibis-framework[duckdb]' # (1) (2)
    ```

    1. We suggest starting with the DuckDB backend.  It's performant and fully featured.  If you would like to use a different backend, all of the available options are listed below.

    2. Note that the `ibis-framework` package is *not* the same as the `ibis` package in PyPI.  These two libraries cannot coexist in the same Python environment, as they are both imported with the `ibis` module name.

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-framework
    ```

{% endfor %}

## Install backend dependencies

{% for backend in sorted(ibis.backends.base._get_backend_names()) %}
=== "{{ backend }}"

    ```sh
    pip install 'ibis-framework[{{ backend }}]'
    ```

{% endfor %}

---

After you've successfully installed Ibis, try going through the tutorial:

<div class="install-tutorial-button" markdown>
[Go to the Tutorial](./tutorial/index.md){ .md-button .md-button--primary }
</div>
