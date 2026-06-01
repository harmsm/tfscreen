========
tfscreen
========

`tfscreen` is a Python library for simulating and analyzing high-throughput screens of transcription
factor (TF) function. It is designed to infer the energetic effects of mutations on conformations
in the TF energy landscape — information that enables interpretable predictive models of how
arbitrary combinations of mutations affect the TF response curve (operator binding versus allosteric
effector).

The core strategy is to measure operator occupancy for thousands of double-mutant cycles across
multiple allosteric effector concentrations. A single set of energetic effects must explain all
measurements simultaneously, which resolves the per-conformation contribution of each mutation and
separates epistasis from structural contacts (cycle-specific) from epistasis arising from
redistribution of the energy landscape (shared across cycles).

Occupancy is inferred from *E. coli* growth in a dual-marker selection scheme: *kanR* (kanamycin
resistance, growth enhanced when expressed) and *pheS** (4-chloro-L-phenylalanine sensitivity,
growth reduced when expressed) respond in opposite directions to a shift in occupancy. Libraries
of TF variants are grown in different effector/selection combinations and variant frequencies are
followed over time by direct sequencing. A hierarchical Bayesian model jointly fits the growth
data and direct binding measurements, learning the relationship between occupancy and growth rate
along the way.

Documentation
-------------

Full documentation is available at https://tfscreen.readthedocs.io.

Installation
------------

Clone the repository and install in editable mode::

    git clone https://github.com/harmslab/tfscreen
    cd tfscreen
    pip install -e .
