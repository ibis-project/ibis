# Contribute to Ibis

{{ config.extra.project_name }} is developed and maintained by a [community of
volunteer contributors]({{ config.repo_url }}/graphs/contributors).

{% for group in config.extra.team %}

## {{ group.name }}

{% for person in group.members %}

- https://github.com/{{ person }}
  {% endfor %}

{% endfor %}
