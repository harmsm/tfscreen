=========
The model
=========

``tfscreen.tfmodel`` is a hierarchical Bayesian model written in JAX and
NumPyro. It infers each genotype's operator occupancy *θ* as a function of
titrant concentration from growth data, binding data or both. This page
describes the generative model and every component that can be swapped into
it. :doc:`fitting` describes how to fit it, and :doc:`model-inputs` describes
the data it reads.

The growth equation
===================

Each genotype *g* grows in a tube at a rate set by its occupancy. With the
default components, linear condition growth and an instant transition between
the pre-growth and selection phases, the predicted abundance is

.. math::

   \ln \mathrm{cfu} = \ln \mathrm{cfu}_0
       + (k_\mathrm{pre} + dk_g + A_g\, m_\mathrm{pre}\, \theta_g)\, t_\mathrm{pre}
       + (k_\mathrm{sel} + dk_g + A_g\, m_\mathrm{sel}\, \theta_g)\, t_\mathrm{sel}
       + \delta_\mathrm{tube}

*k* and *m* are the baseline growth rate and the occupancy slope of each
condition, shared by every genotype. *dk_g* is the genotype's pleiotropic
growth effect, independent of the transcription factor. *A_g* is its TF
activity. *θ_g* is its occupancy at the tube's titrant concentration.
*δ_tube* is an offset shared by every genotype in one tube. Growth rates are
per minute and times are in minutes.

Each term comes from a component chosen at configuration time. The
``condition_growth`` component sets how *k*, *m* and *θ* combine, the
``growth_transition`` component sets how the two phases join, and so on down
the list below. With the ``mixture`` transformation each genotype's cells are
split into a clean class and several congressed classes, each grown through
the same equation, and the classes are mixed as

.. math::

   \ln \mathrm{cfu} = \ln \mathrm{cfu}_0
       + \log \sum_c w_c \exp(G_c) + \delta_\mathrm{tube}

where *G_c* is the growth term of class *c* and *w_c* its weight.

The predicted ``ln_cfu`` is observed through the growth likelihood. Binding
data, when supplied, observe *θ* directly. Pre-split and base-growth data,
when supplied, add their own observations of the starting abundance and of
reference growth rates.

.. _model-components:

Components
==========

``tfs-configure-model`` selects one component per axis with a
``--<axis>_model`` flag and writes the choice into the ``components:`` block
of the configuration YAML. Every option, with its default, is in
:doc:`cli`. The defaults are the current recommendations for a joint
growth and binding model.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Flag
     - Default
     - Options
   * - ``--condition_growth_model``
     - ``linear``
     - ``linear``, ``power``, ``saturation``
   * - ``--growth_transition_model``
     - ``instant``
     - ``instant``, ``memory``, ``baranyi``, ``baranyi_k``, ``baranyi_tau``,
       ``two_pop``
   * - ``--ln_cfu0_model``
     - ``hierarchical``
     - ``hierarchical``, ``hierarchical_factored``
   * - ``--dk_geno_model``
     - ``hierarchical_geno``
     - ``hierarchical_geno``, ``fixed``, ``pinned``
   * - ``--activity_model``
     - ``fixed``
     - ``fixed``, ``hierarchical_geno``, ``horseshoe_geno``,
       ``hierarchical_mut``, ``horseshoe_mut``
   * - ``--theta_model``
     - ``hill_geno``
     - ``hill_geno``, ``hill_relative``, ``hill_mut``, ``categorical_geno``,
       ``thermo.*``
   * - ``--theta_rescale_model``
     - ``passthrough``
     - ``passthrough``, ``logit``
   * - ``--transformation_model``
     - ``single``
     - ``single``, ``mixture``
   * - ``--theta_growth_noise_model``
     - ``zero``
     - ``zero``, ``beta``, ``logit_normal``
   * - ``--theta_binding_noise_model``
     - ``zero``
     - ``zero``, ``beta``
   * - ``--growth_noise_model``
     - ``zero``
     - ``zero``, ``normal_kt``
   * - ``--sample_offset_model``
     - ``level``
     - ``level``, ``zero``, ``normal``
   * - ``--growth_likelihood``
     - ``counts``
     - ``counts``, ``lncfu``

Each component's priors are written to ``{out_prefix}_priors.csv`` with rows
named ``{group}.{component}.{field}``, such as
``growth.sample_offset.sigma_fixed``. Change a scalar prior with
``tfs-configure-model --set_priors name=value`` rather than by editing the
file. A name can be the full row name or any unique dotted suffix of it.

Condition growth
----------------

The ``condition_growth`` component maps occupancy to a growth rate in each
condition. A condition is one marker and selection combination, such as
``kanR+kan`` or ``pheS+4CP``.

``linear`` (default)
   *g = k + dk_g + A·m·θ*. One *k* and one *m* per condition. Control
   conditions, those without selection, get a tight prior on *m* set by
   ``m_scale_minus``, default 0.001. Selection conditions get a looser one
   set by ``m_scale_plus``, default 0.01.
``power``
   *g = k + dk_g + A·m·θ*\ :sup:`n`, with a per-condition exponent *n*.
``saturation``
   *g = min + dk_g + A·(max − min)·θ/(1 + θ)*, with per-condition minimum and
   maximum rates.

``power`` and ``saturation`` assume *θ* lies in [0, 1], so pair them with the
``passthrough`` rescale. ``--growth_shares_replicates`` gives every replicate
the same condition parameters. Without it each replicate has its own.

The baseline *k* is only identified jointly with *dk_g*: adding a constant to
every *k* and subtracting it from every *dk_g* leaves the likelihood
unchanged. Wt's *dk_g* is fixed at 0, but that single anchor is diluted by the
size of the library, and the whole system can slide. The fix is a
per-condition prior on the baselines. Every prior location and scale of these
components accepts a scalar, applied to every condition, or one value per
condition. ``tfs-prefit-calibration`` writes per-condition *k* and *m*
locations into the priors file, as indexed rows labeled by ``condition_rep``,
with a tight floored scale. Its ``--pin_m`` option clamps *m* to the
calibrated value outright, because a soft prior on *m* can be overridden by the
growth likelihood. *k* keeps a soft prior, since it carries real
tube-to-tube variation. For a growth-only model, where the pre-fit does not
run, set the per-condition priors at configuration with ``--growth_priors``
or ``--growth_priors_wt_rates``. See :doc:`fitting`.

Growth transition
-----------------

The ``growth_transition`` component joins the pre-growth and selection
phases.

``instant`` (default)
   Each genotype switches to its selection rate at the start of selection.
``memory``
   Cells keep the pre-growth rate for a lag *τ = τ0 + k1/(θ + k2)*, then grow
   at the selection rate. The lag depends on occupancy.
``baranyi``
   The rate moves from the pre-growth to the selection rate along a sigmoid in
   time, with a per-condition midpoint *τ* and steepness.
``baranyi_k``
   As ``baranyi``, with the steepness reduced as the rate change grows.
``baranyi_tau``
   As ``baranyi``, with the midpoint delayed as the rate change grows.
``two_pop``
   Cells leave the pre-growth state at a constant per-cell rate, so the
   culture is a mix of the two growth modes during the transition.

Starting abundance
------------------

The ``ln_cfu0`` component sets each genotype's starting abundance.

``hierarchical`` (default, recommended)
   One starting abundance per replicate, pre-growth condition and genotype.
   Each library class has its own pooled prior, and spiked genotypes and wt
   have their own priors.
``hierarchical_factored``
   One genotype baseline per replicate, shared by every pre-growth
   condition, plus one offset per tube. This is valid only when the
   pre-growth conditions are split from one culture. In the screen, the kanR
   and pheS libraries are transformed and grown separately, so each
   genotype's starting abundance differs between them. On simulations of that
   design the factored model pushed the difference into a confident
   per-genotype error in *θ*, with 95% coverage of 0.09 above 1000 reads
   against 0.80 for ``hierarchical``. ``tfs-configure-model`` refuses
   ``hierarchical_factored`` when the growth table's ``library`` column shows
   a replicate's pre-growth conditions come from different libraries.

Pleiotropic growth effect
-------------------------

The ``dk_geno`` component sets *dk_g*, each genotype's effect on growth that
does not go through the transcription factor.

``hierarchical_geno`` (default)
   A pooled, left-skewed prior: a shift minus a log-normal draw. Most
   mutations are near neutral, a few help, and a long tail is deleterious.
   Wt is fixed at 0.
``fixed``
   *dk_g* = 0 for every genotype.
``pinned``
   *dk_g* is fixed to supplied values for a subset of genotypes and 0 for the
   rest. The values come from a CSV given as the orchestrator's
   ``dk_geno_pins_file`` setting. ``tfs-configure-model`` has no flag for that
   file, so this component is only reachable by building a
   ``ModelOrchestrator`` in Python.

Activity
--------

The ``activity`` component sets *A_g*, a factor on each genotype's occupancy
term.

``fixed`` (default)
   *A* = 1 for every genotype.
``hierarchical_geno``
   One activity per genotype from a pooled log-normal prior.
``horseshoe_geno``
   One activity per genotype under a sparse horseshoe prior.
``hierarchical_mut``
   Log activity is a sum of per-mutation effects, plus pairwise epistasis with
   ``--epistasis``, under hierarchical normal priors.
``horseshoe_mut``
   As ``hierarchical_mut``, under regularized horseshoe priors.

A learned activity is refused with ``hill_relative`` and with the
``mixture`` transformation's ``homodimer`` and ``heterodimer`` rules.

Occupancy
---------

The ``theta`` component sets *θ* as a function of titrant concentration *c*.

``hill_geno`` (default)
   A Hill curve per genotype,
   *θ = θ_low + (θ_high − θ_low)·sigmoid(n·(ln c − ln K))*. The parameters
   are ``theta_low``, ``theta_high``, ``log_hill_K`` and ``hill_n``.
   ``log_hill_K`` is the natural log of the midpoint concentration. Each
   parameter is drawn from a population distribution whose location and
   scale are learned.
``hill_relative``
   The Hill curve for a growth-only fit on a wt-relative scale *X* in place
   of *θ*. See below.
``hill_mut``
   A Hill curve whose transformed parameters are wt's values plus a sum of
   per-mutation effects, plus pairwise epistasis under a regularized
   horseshoe prior when ``--epistasis`` is set. It can predict genotypes that
   were never measured.
``categorical_geno``
   An independent *θ* at every titrant concentration, with no curve.
``thermo.*``
   Thermodynamic partition-function models. See
   :ref:`thermo-naming` below. They need ``--thermo_data``.

The population SD of each Hill parameter is learned by default. On real
data a learned SD can run away and stop shrinking poorly measured genotypes:
a MAP on the development screen drove the SD of log(hill_n) to about 19, and
*n* near 0.01 hid each genotype's response below the lowest nonzero
concentration. ``theta_log_hill_n_hyper_scale_fixed`` holds that SD at a
given value instead. A value of 0, the default, keeps it learned.

.. code-block:: bash

   tfs-configure-model ... --set_priors theta_log_hill_n_hyper_scale_fixed=0.5

The relative-X fit
~~~~~~~~~~~~~~~~~~

Growth alone fixes an occupancy-like variable only up to an affine map,
because *k* and *m* absorb any shift and scale. ``hill_relative`` therefore
fits a wt-relative variable *X* with the Hill curve's form and real-valued
baselines, ``X_low`` and ``X_delta``. Wt is gauged to *X* = 1 at a low
concentration and *X* = 0 at a high one. ``--theta_gauge_conc c_lo c_hi``
sets those two concentrations, by default the lowest and highest in the
growth table. The resolved values are written to the configuration.

``hill_relative`` is refused with binding data, with an activity other than
``fixed``, a rescale other than ``passthrough``, condition growth other than
``linear``, theta growth noise other than ``zero`` and, under the ``mixture``
transformation, any congression *θ* rule but ``max``. Downstream tools know
the output is *X*: ``tfs-predict-theta`` adds a ``theta_scale`` column set
to ``X``, and epistasis on *X* is defined only on the additive scale.

Besides ``theta_log_hill_n_hyper_scale_fixed``, ``hill_relative`` takes
``theta_X_low_hyper_scale_fixed``, ``theta_X_delta_hyper_scale_fixed`` and
``theta_log_hill_K_hyper_scale_fixed``. Learned, these ran to 3.6, 4.3 and 11
on the development screen, against about 0.5, 0.5 and 0.7 among
well-measured genotypes, so the fit shrank nothing.

Occupancy rescale
-----------------

The ``theta_rescale`` component transforms *θ* before it enters condition
growth.

``passthrough`` (default)
   *θ* is used as is.
``logit``
   *θ* is replaced by log(*θ*/(1 − *θ*)), which expands the range near 0
   and 1.

Congression
-----------

The ``transformation`` component models congression, more than one plasmid
entering a cell during transformation.

``single`` (default)
   Every cell carries one plasmid.
``mixture``
   Each genotype's cells are a mixture of clean cells and congressed cells
   that also carry plasmids drawn from the bulk library. Transformants carry
   a zero-truncated Poisson(λ) number of plasmids. A genotype's congressed
   fraction is its ``bulk_fraction`` from the library table times
   1 − P(M = 1). Each class has its own cell-level *θ*, activity and
   *dk_g*, grows through the full equation, and the classes are mixed at the
   level of ``exp(ln_cfu)``.

``mixture`` needs ``--transformation_lambda MEAN SD``, the measured
congression rate in linear space, which becomes a log-normal prior on λ. It
is refused with ``single``. Three settings fix how a congressed cell combines
its plasmids:

``--congression_theta_rule``
   ``homodimer`` (default) combines the plasmids' occupancies through the
   partition function of a homodimer with equal shares, ``heterodimer``
   through that of a heterodimer, and ``max`` takes the highest *θ*. The
   ``homodimer`` and ``heterodimer`` rules require ``activity`` ``fixed``.
``--congression_dk_rule``
   ``dilution`` (default) takes the share-weighted mean of *dk_g*, ``min``
   the worst variant's, and ``softmin`` interpolates between them with
   ``--congression_dk_alpha``.

The congressed term is averaged over fixed co-resident sets drawn once per
genotype from the bulk library. ``congression_sets`` sets how many sets of
each co-resident count, default ``[12, 3, 1]`` for one, two and three
co-residents, and ``congression_seed`` (default 0) makes the draw
reproducible. ``tfs-configure-model`` writes both into the configuration's
``components:`` block and has no flag for them.

The transformations ``empirical`` and ``logit_norm`` were removed. A
configuration that names them is refused with a message.

Noise
-----

``--theta_growth_noise_model`` adds noise to the *θ* that enters growth, and
``--theta_binding_noise_model`` to the *θ* that binding observes.

``zero`` (default)
   No noise.
``beta``
   *θ* is drawn from a beta distribution around the predicted value.
``logit_normal``
   Normal noise with a learned SD is added to logit(*θ*). Growth only.

``--growth_noise_model`` adds noise to growth itself.

``zero`` (default)
   No extra noise.
``normal_kt``
   A learned global SD on accumulated growth, added in quadrature to
   ``ln_cfu_std``. It works only with the ``lncfu`` likelihood.

Tube offsets
------------

The ``sample_offset`` component adds *δ_tube*, a shift shared by every
genotype in one tube. A tube is one sequenced sample: one replicate, time,
condition and titrant concentration. Anything that moves every genotype in a
tube together lands here, such as PCR and genotype-calling efficiency or an
error in the tube's total cell count.

``level`` (default)
   One ``ln_cfu`` offset per tube with a learned SD. ``sigma_fixed`` > 0
   holds the SD at that value. Learned, the SD grew from its 0.2 prior scale
   to 0.53 on the first real-data fit, and the offsets took over the
   population's growth from *k* and *m*. Holding it near the per-tube OD600
   scatter, about 0.17, keeps the tube totals binding:
   ``--set_priors sigma_fixed=0.17``.
``zero``
   No offset.
``normal``
   One growth-rate offset per tube, multiplied by the tube's elapsed time.

Observations
============

Growth likelihood
-----------------

``--growth_likelihood`` sets how growth data are observed.

``counts`` (default)
   Reads are observed directly as negative binomial,
   *reads ~ NegBin(μ, var = μ(1 + φ) + μ²/r)*, where
   *log μ = ln(sample reads) + ln_cfu_pred − ln(sample cfu)*. That is the
   predicted frequency times the tube's total reads. φ and 1/r are learned
   with log-normal priors. No pseudocount is used. It needs a ``counts``
   column, each tube's total reads and each tube's total cells, which
   ``tfs-process-counts`` writes. It refuses a growth noise other than
   ``zero``. Pair it with ``level`` tube offsets, since an error in a tube's
   total otherwise moves every genotype in it. On simulations it cut *θ*
   RMSE by 35-60% against ``lncfu`` at similar coverage.
``lncfu``
   Student-t on ``ln_cfu`` with the table's ``ln_cfu_std``.

``ModelOrchestrator`` itself still defaults to ``lncfu`` and no tube
offset, so that configurations written before the count likelihood existed
read back unchanged. ``tfs-configure-model`` writes the new defaults
explicitly.

Other observations
------------------

Binding data, from ``--binding_df``, observe *θ* with a normal likelihood.
Because growth rows usually outnumber binding rows by orders of magnitude,
the binding likelihood is scaled by ``--binding_weight``, by default the
ratio of growth rows to binding rows. A model without binding data has no
binding sites at all.

Pre-split data, from ``--presplit_df``, observe ``ln_cfu`` at the start of
pre-growth and constrain the starting abundance. Base-growth data, from
``--base_growth_df``, observe reference-condition growth rates as
*rate ~ Normal(k_ref + dk_g, rate_std)* and anchor *dk_g*. See
:doc:`model-inputs` for both.

.. _model-naming:

Naming conventions
==================

Components that infer one value per genotype carry a ``_geno`` suffix, such
as ``hill_geno``. Components that build each genotype from per-mutation
effects carry ``_mut``, such as ``hill_mut``. Components with no natural
per-mutation form, such as ``fixed``, have no suffix.

.. _thermo-naming:

Thermodynamic occupancy models
------------------------------

Occupancy models built from an explicit partition function are named
``thermo.{MODEL}.{PRIOR}``. *MODEL* has four fields and a letter:

.. list-table::
   :header-rows: 1
   :widths: 10 60

   * - Field
     - Meaning
   * - ``O``
     - Oligomeric state; ``O2`` is a homodimer
   * - ``C``
     - Number of conformational states
   * - ``K``
     - Number of independent equilibrium constants
   * - ``U``
     - ``U0`` has folded states only; ``U1`` adds a folding equilibrium

The trailing letter tells apart models with the same counts and implies no
order. The implemented models are:

* ``O2_C4_K3_U0_a``: four-state lac repressor homodimer.
* ``O2_C4_K3_U1_a``: the same with a folding equilibrium.
* ``O2_C12_K5_U0_a``: two-state MWC homodimer.
* ``O2_C12_K5_U1_a``: the same with a folding equilibrium.

*PRIOR* sets how the equilibrium constants are parameterized:

.. list-table::
   :header-rows: 1
   :widths: 10 60

   * - Name
     - Description
   * - ``PK``
     - An independent normal prior on each log *K*, built from per-mutation
       effects.
   * - ``PddG``
     - Priors centered on supplied ΔΔG estimates, a CSV given with
       ``--thermo_data`` with one column per structure.
   * - ``PnnC``
     - A neural network that predicts per-conformation ΔΔG from structural
       features, an HDF5 file given with ``--thermo_data``. See
       :doc:`ligandmpnn-features`.

Full names look like ``thermo.O2_C4_K3_U0_a.PK`` or
``thermo.O2_C12_K5_U1_a.PnnC``.
