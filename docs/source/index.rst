========
tfscreen
========

`tfscreen` is a Python library for simulating and analyzing high-throughput screens of transcription factor function.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   simulation
   process-raw
   analysis

Simulation
----------
`tfscreen` allows you to simulate high-throughput screens starting from thermodynamic models of transcription factor binding and activity. See the :doc:`simulation` page for more details.

Raw Data Processing
-------------------
`tfscreen` includes utilities to convert raw sequencing data into quantitative inputs for analysis. See the :doc:`process-raw` page for more details.

Analysis
--------
`tfscreen` provides robust statistical tools, including Bayesian hierarchical models, to extract biochemical parameters from screen data. See the :doc:`analysis` page for more details.

Nomenclature
----------
* **condition**: A growth condition defined by marker, selection, and iptg. A genotype will have the same average growth rate in the same condition.
* **sample**: A tube growing under a specific growth condition. It is defined by replicate, marker, selection, and iptg.
* **timepoint**: An aliquot of a given sample taken at a specific time. It is defined by replicate, marker, selection, iptg, and time.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
