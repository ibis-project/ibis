# Getting started

## Installation instructions

The next steps provides the easiest and recommended way to set up your
environment to use {{ ibis.project_name }}. Other installation options can be found in
the [advanced installation page]({{ base_url}}/docs/index.html#installation).

1. Download [Anaconda](https://www.anaconda.com/distribution/) for your operating system and
   the latest Python version, run the installer, and follow the steps. Detailed instructions
   on how to install Anaconda can be found in the
   [Anaconda documentation](https://docs.anaconda.com/anaconda/install/)).

2. In the Anaconda prompt (or terminal in Linux or MacOS), install {{ ibis.project_name }}:

        :::sh
        conda install -c conda-forge ibis-framework

3. In the Anaconda prompt (or terminal in Linux or MacOS), start JupyterLab:

    <img class="img-fluid" alt="" src="{{ base_url }}/static/img/install/anaconda_prompt.png"/>

4. In JupyterLab, create a new (Python 3) notebook:

    <img class="img-fluid" alt="" src="{{ base_url }}/static/img/install/jupyterlab_home.png"/>

5. In the first cell of the notebook, you can import {{ ibis.project_name }} and check the version with:

        :::python
        import ibis
        ibis.__version__

6. Now you are ready to use {{ ibis.project_name }}, and you can write your code in the next cells.

## Tutorials

You can learn more about {{ ibis.project_name }} in the
[tutorials](https://ibis-project.org/docs/tutorial/index.html),
and more about JupyterLab in the [JupyterLab documentation](https://jupyterlab.readthedocs.io/en/stable/user/interface.html).
