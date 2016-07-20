import os
import shutil
import subprocess
import glob


def rstify_notebook(path, outpath):
    cmd = ('jupyter nbconvert --execute --to=rst {0} --output {1}'
           .format(path, outpath))

    print cmd
    subprocess.check_call(cmd, shell=True)

path = '../../ibis-notebooks/basic-tutorial'
notebooks = sorted(glob.glob(os.path.join(path, '*.ipynb')))

outdir = 'source/generated-notebooks'
if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.mkdir(outdir)

with open(os.path.join(outdir, 'manifest.txt'), 'w') as manifest:
    for i, path in enumerate(notebooks):
        base, tail = os.path.split(path)

        root, _ = os.path.splitext(tail)

        fpath = '{0}.rst'.format(i)
        outpath = os.path.join(outdir, fpath)
        rstify_notebook(path, outpath)

        stub = """\
Notebook {0}

.. toctree::
   :maxdepth: 1

   generated-notebooks/{1}

""".format(i + 1, i)

        manifest.write(stub)
