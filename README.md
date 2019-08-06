# nawrapper

`nawrapper` is a thin wrapper around [NaMaster](https://github.com/LSSTDESC/NaMaster) for CMB work, particularly for the ACT experiment. The intention is to hide some of NaMaster's generality for conciseness and simplicity.

Read the [documentation](http://physics.princeton.edu/~zequnl/nawrapper/docs/build/html/index.html)!

## Installation
This guide is intended for people who are not currently using Anaconda on Niagara. If you already have a conda environment on Niagara which works with Jupyter, skip to the next section titled "Installing the Power Spectrum Software".

### Create a Conda Environment
You first need to get the Anaconda module and enter a custom environment.

```bash
module load anaconda3
```

Either enter your preferred conda environment with `source activate YOUR_ENV_NAME` if you already have one, or create a new one. I call my power spectrum conda environment `ps`, but you can name it something else if you want, just replace every mention of `ps` in the instructions below with your desired environment name.

```bash
conda create -n ps python=3
```

Enter your environment. This is what you will type every time you want to compute some spectra (and in your SLURM power spectrum jobs!).
```bash
conda activate ps
```

You will need to install some other necessary packages into your conda environment, if you just created it. Note that we are using `-n` to specify the conda environment!

```bash
conda install -n ps matplotlib ipykernel cython numpy scipy astropy pillow
```

If you are running on a machine with Jupyter like cori or niagara, we also set up ipykernel so your conda environment shows up in Jupyter. You can do this on tiger too, but it's less convenient (see [Jupyter on the Cluster](https://oncomputingwell.princeton.edu/2018/05/jupyter-on-the-cluster/)).
```bash
python -m ipykernel install --user --name ps --display-name "Python 3 (ps)"
```

### Installing the Power Spectrum Software

Next, install NaMaster with
```bash
conda install -c conda-forge namaster -n ps
```

You must also install pixell if you do not have it.

```bash
module load autotools
git clone git@github.com:simonsobs/pixell.git
cd pixell
python setup.py install --user
```

Then install nawrapper (my routines for ACT power spectrum analysis).

```bash
git clone git@github.com:xzackli/nawrapper.git
cd nawrapper
pip install -e . --user
```

You should be all set! Try out the `Getting Started.ipynb` in the `notebooks/` folder.
