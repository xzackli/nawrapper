.. NaWrapper documentation master file, created by
   sphinx-quickstart on Tue Aug  6 11:23:51 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NaWrapper
=========

`NaWrapper`_ is a thin python wrapper around `NaMaster`_ for CMB work,
in particular for the ACT experiment.

.. figure:: _static/outline.svg

It bundles all the necessary pieces associated with a map (i.e. beams, masks,
filters) into a :py:class:`nawrapper.ps.namap` object, and provides concise
utility functions for turning these into dictionaries of power spectra.


.. toctree::
   :maxdepth: 2

   usage/installation
   usage/quickstart
   usage/examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _NaMaster: https://github.com/LSSTDESC/NaMaster
.. _nawrapper: https://github.com/xzackli/nawrapper
