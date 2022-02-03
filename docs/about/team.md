# Team

## Contributors

{{ config.extra.project_name }} is developed and maintained by a [community of
volunteer contributors]({{ config.repo_url }}graphs/contributors).

{% for group in config.extra.team %}

## {{ group.name }}

{% for person in group.members %}

- https://github.com/{{ person }}
  {% endfor %}

{% endfor %}

{{ config.extra.project_name }} aims to be a welcoming, friendly, diverse and
inclusive community. Everybody is welcome, regardless of gender, sexual
orientation, gender identity, and expression, disability, physical appearance,
body size, race, or religion. We do not tolerate harassment of community
members in any form. In particular, people from underrepresented groups are
encouraged to join the community.
