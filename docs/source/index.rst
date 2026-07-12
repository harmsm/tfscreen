========
tfscreen
========

`tfscreen` is a Python library for simulating and analyzing high-throughput screens of transcription factor function.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   simulation
   process-raw
   model-inputs
   analysis
   summarize-fit
   grid
   ligandmpnn-features

Goal
----

At the highest level, `tfscreen` is designed to infer the energetic effects of mutations on different conformations in the energy landscape of bacterial transcription factors. The hypothesis is that learning these energetic effects will make it possible to build interpretable predictive models that accept any arbitrary combination of mutations and predict the resulting TF response curve (operator binding versus allosteric effector). The long-term goal is to perform this as a single inference, with the model ingesting experimental training data and returning thermodynamic parameters.

The core idea is to use many slightly different measurements of protein function to reveal the energy landscape. The approach starts by writing out a thermodynamic model that describes the conformations in the energy landscape, then measuring occupancy for thousands of pairwise mutant cycles at multiple allosteric effector concentrations. Because a single consistent set of energetic effects must explain measurements spread across many effector concentrations and genetic backgrounds, this resolves the additive energetic contribution of each mutation to each conformation. It also cleanly distinguishes between two sources of epistasis: epistasis arising from direct structural contacts between residues (unique to each cycle) versus epistasis arising from redistribution of the energy landscape (shared across all cycles with those mutations).

Approach
--------

Experimentally, `tfscreen` uses a strategy combining commercial oligo pools with Golden Gate cloning to generate defined libraries of single mutants and double mutants. This produces libraries with precisely defined compositions that consist entirely of double-mutant cycles. In the first experiment, approximately 200,000 mutant cycles were measured.

The thermodynamic observable is operator occupancy (*θ*), inferred from *E. coli* growth using a dual-marker selection scheme. The TF operator is placed upstream of two conditional markers with inverted logic: *kanR* confers kanamycin resistance, so growth is *enhanced* when it is expressed; *pheS** confers 4-chloro-L-phenylalanine sensitivity, so growth is *reduced* when it is expressed. A shift in occupancy therefore produces opposing shifts in growth rate for the two selection conditions. This signal is detected by growing a library of TF variants in different effector/selection combinations, then following variant frequency over time by direct sequencing.

The model in ``tfscreen.tfmodel`` is designed to jointly capture high-throughput growth data and low-throughput binding data, learning the quantitative relationship between operator occupancy and growth rate in the process.

Quick Start
-----------
New to ``tfscreen``?  The :doc:`quickstart` page walks through a complete
simulate-and-analyze run using the bundled example in
``examples/simulate-and-analyze/``.

Simulation
----------
`tfscreen` allows you to simulate high-throughput screens starting from thermodynamic models of transcription factor binding and activity. See the :doc:`simulation` page for more details.

Raw Data Processing
-------------------
`tfscreen` includes utilities to convert raw sequencing data into quantitative inputs for analysis. See the :doc:`process-raw` page for more details.

Model Input Data
-----------------
The hierarchical model accepts binding, growth, base-growth, pre-split, and congression-calibration data. See the :doc:`model-inputs` page for what each contributes, how much is useful, and how to pass it in.

Analysis
--------
`tfscreen` provides robust statistical tools, including Bayesian hierarchical models, to extract biochemical parameters from screen data. See the :doc:`analysis` page for more details.

Grid Setup
----------
Both the simulation and model-fitting workflows support parameter sweeps via a grid mechanism. See the :doc:`grid` page for the grid YAML format and the ``tfs-setup-grid``, ``tfs-setup-sim-grid``, and ``tfs-summarize-grid`` commands.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
