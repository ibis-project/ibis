# Team

## Contributors

{{ main.project_name }} is developed and maintained by a
[community of volunteer contributors](https://github.com/{{ main.github_repo_url }}/graphs/contributors).

If you want to support pandas development, you can find information in the [donations page](../donate.html).

## Maintainers

<div class="row maintainers">
    {% for row in maintainers.people | batch(6, "") %}
        <div class="card-group maintainers">
            {% for person in row %}
                {% if person %}
                    <div class="card">
                        <img class="card-img-top" alt="" src="{{ person.avatar_url }}"/>
                        <div class="card-body">
                            <h6 class="card-title">
                                {% if person.blog %}
                                    <a href="{{ person.blog }}">
                                        {{ person.name or person.login }}
                                    </a>
                                {% else %}
                                    {{ person.name or person.login }}
                                {% endif %}
                            </h6>
                            <p class="card-text small"><a href="{{ person.html_url }}">{{ person.login }}</a></p>
                        </div>
                    </div>
                {% else %}
                    <div class="card border-0"></div>
                {% endif %}
            {% endfor %}
        </div>
    {% endfor %}
</div>

{{ main.project_name }} aims to be a welcoming, friendly, diverse and inclusive community.
Everybody is welcome, regardless of gender, sexual orientation, gender identity,
and expression, disability, physical appearance, body size, race, or religion.
We do not tolerate harassment of community members in any form.
In particular, people from underrepresented groups are encouraged to join the community.

{% if maintainers.emeritus %}
## Emeritus maintainers

<ul>
    {% for person in maintainers.emeritus %}
        <li>{{ person }}</li>
    {% endfor %}
</ul>
{% endif %}
