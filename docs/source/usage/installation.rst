Installation
============


This guide is intended for people who are not currently using Anaconda. If you
already have a Python 3 conda environment, skip below to the
subsection :ref:`installingsoftware`.

Create a Conda Environment
--------------------------

You first need to get the Anaconda module and enter a custom
environment.

.. code:: bash

   module load anaconda3

Either enter your preferred conda environment with
``source activate YOUR_ENV_NAME`` if you already have one, or create a
new one. I call my power spectrum conda environment ``ps``, but you can
name it something else if you want, just replace every mention of ``ps``
in the instructions below with your desired environment name.

.. code:: bash

   conda create -n ps python=3

Enter your environment. This is what you will type every time you want
to compute some spectra (and in your SLURM power spectrum jobs!).

.. code:: bash

  conda activate ps

You will need to install some other necessary packages into your conda
environment, if you just created it. Note that we are using ``-n`` to
specify the conda environment!

.. code:: bash

  conda install -n ps matplotlib cython numpy scipy astropy pillow

If you are running on a machine with Jupyter like cori or niagara, we
also set up ipykernel so your conda environment shows up in Jupyter. You
can do this on tiger too, but it’s less convenient (see `Jupyter on the
Cluster`_).

.. code:: bash

  conda install -n ps ipykernel
  python -m ipykernel install --user \
     --name ps --display-name "Python 3 (ps)"


.. _installingsoftware:

Installing Software
-------------------

We now install the power spectrum software. Install NaMaster with

.. code:: bash

  conda install -c conda-forge namaster -n ps

You also need to install pixell for CAR map manipulation. This involves cloning
a GitHub repository, so for convenience you might want to `set up your SSH keys
on GitHub`_. If you are on niagara, you will also need to
``module load autotools`` to get autoconf
before installing pixell. Go to the directory where you keep your software, and
run

.. code:: bash

  git clone git@github.com:simonsobs/pixell.git
  cd pixell
  python setup.py install --user

You also need `healpy` to manipulate healpix maps.

.. code:: bash

    pip install healpy

Then install nawrapper (these routines for ACT power spectrum analysis). Return to
the directory where you keep your software, and then run

.. code:: bash

  git clone git@github.com:xzackli/nawrapper.git
  cd nawrapper
  pip install -e . --user

You should be all set! Try out the :ref:`quickstart`.



Updating NaWrapper
~~~~~~~~~~~~~~~~~~
To get new version of NaWrapper in the future, you'll have to run `git pull` in
the `nawrapper` repository you had cloned earlier. Since you installed with `-e`
using pip, this is all you have to do.



.. _NaMaster: https://github.com/LSSTDESC/NaMaster
.. _documentation: http://physics.princeton.edu/~zequnl/nawrapper/docs/build/html/index.html
.. _Jupyter on the Cluster: https://oncomputingwell.princeton.edu/2018/05/jupyter-on-the-cluster/
.. _set up your SSH keys on GitHub: https://help.github.com/en/articles/adding-a-new-ssh-key-to-your-github-account
