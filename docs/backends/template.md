# [{{ backend_name }}]({{ backend_url }})

{% if is_experimental %}
!!! experimental

    The {{ backend_name }} is experimental and is subject to breaking changes.

{% endif %}

## Install

Install dependencies for Ibis's {{ backend_name }} backend:

=== "pip"

    ```sh
    pip install 'ibis-framework[{{ backend_module }}]'
    ```

=== "conda"

    ```sh
    conda install -c conda-forge ibis-{{ backend_module }}
    ```

## Connect

<!-- prettier-ignore-start -->
Create a client by passing in {{ backend_param_style }} to [`ibis.{{ backend_module }}.connect`][ibis.backends.{{ backend_module }}.Backend.do_connect]:
<!-- prettier-ignore-end -->

```python
{{ backend_connection_example }}
```
