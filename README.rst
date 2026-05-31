========
tfscreen
========

Library for simulating and analyzing high-throughput screens of transcription
factor function.

Nomenclature
------------
+ *condition*: A growth condition defined by marker, selection, and iptg. A
  genotype will have the same average growth rate in the same condition.
+ *sample*: A tube growing under a specific growth condition. It is defined by
  replicate, marker, selection, and iptg.
+ *timepoint*: An aliquot of a given sample taken at a specific time. It is
  defined by replicate, marker, selection, iptg, and time.

Model naming conventions
------------------------

Model components are selected by name in the run configuration YAML.  The
names follow a set of conventions that encode what each component does.

**Level of parameterization**

Models that infer one parameter per *genotype* carry a ``_geno`` suffix;
models that decompose effects at the *mutation* level carry a ``_mut`` suffix.
Models with no natural per-mutation alternative (e.g. ``fixed``) have no
suffix.

Examples: ``hierarchical_geno``, ``horseshoe_mut``, ``hill_geno``,
``hill_mut``.

**Thermodynamic theta models**

Operator-occupancy (θ) models derived from an explicit partition function use
a three-part dot-separated name::

    thermo.{MODEL}.{PRIOR}

*MODEL* encodes the partition-function topology with four fields separated by
underscores:

+-------+--------------------------------------------------------------------+
| Field | Meaning                                                            |
+=======+====================================================================+
| ``O`` | Oligomeric state (e.g. ``O2`` = homodimer)                         |
+-------+--------------------------------------------------------------------+
| ``C`` | Number of conformational states                                     |
+-------+--------------------------------------------------------------------+
| ``K`` | Number of independent equilibrium constants                        |
+-------+--------------------------------------------------------------------+
| ``U`` | Unfolded state: ``U0`` = folded only, ``U1`` = folding equilibrium |
+-------+--------------------------------------------------------------------+

A trailing letter (``a``, ``b``, …) disambiguates topologically distinct
models that share the same O/C/K/U counts.  These letters carry no ordering;
``a`` simply means "first registered variant."

Currently implemented models:

+ ``O2_C4_K3_U0_a`` — four-state lac-repressor homodimer (no unfolding)
+ ``O2_C4_K3_U1_a`` — same with an explicit folding/unfolding equilibrium
+ ``O2_C12_K5_U0_a`` — full MWC two-state homodimer (no unfolding)
+ ``O2_C12_K5_U1_a`` — same with an explicit folding/unfolding equilibrium

*PRIOR* describes how the equilibrium constants are parameterized:

+----------+--------------------------------------------------------------+
| Name     | Description                                                  |
+==========+==============================================================+
| ``PK``   | Independent normal prior on each log-K                       |
+----------+--------------------------------------------------------------+
| ``PddG`` | Priors informed by estimated ΔΔG values                      |
+----------+--------------------------------------------------------------+
| ``PnnC`` | Neural network predicting per-conformation ΔΔG values        |
+----------+--------------------------------------------------------------+
| ``PnnK`` | Neural network predicting log-K values directly (planned)    |
+----------+--------------------------------------------------------------+

Full example names: ``thermo.O2_C4_K3_U0_a.PK``,
``thermo.O2_C12_K5_U1_a.PnnC``.