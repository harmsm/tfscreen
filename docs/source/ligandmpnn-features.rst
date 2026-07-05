===================
LigandMPNN Features
===================

Some tfscreen theta components (the ``PnnC`` prior variants, e.g.
``thermo.O2_C4_K3_U0_a.PnnC``, ``thermo.O2_C4_K3_U1_a.PnnC``,
``thermo.O2_C12_K5_U0_a.PnnC``, ``thermo.O2_C12_K5_U1_a.PnnC``) use per-residue
log-probability features from `LigandMPNN
<https://github.com/dauparas/LigandMPNN>`_, together with a Cα contact-distance
map, as structural inputs to a neural-network-informed ΔΔG prior. These
features are computed once — outside the main tfscreen environment — and
stored in an HDF5 file that is read at analysis time.

.. warning::

   The LigandMPNN conda environment is **incompatible** with the main tfscreen
   environment.  LigandMPNN pins NumPy 1.23.5 and PyTorch, which conflict with
   the JAX/NumPyro stack that the rest of tfscreen requires.  The feature
   generator script is intentionally standalone (no tfscreen installation
   needed) so it can run in the LigandMPNN environment without conflict.  Do
   not attempt to run any other tfscreen commands from the LigandMPNN
   environment.

Overview
--------

The feature generation pipeline is a two-step, two-environment workflow:

1. **LigandMPNN environment** — run ``scripts/generate_struct_ensemble.py`` to
   score each PDB structure and write an HDF5 feature file.
2. **tfscreen environment** — run the hierarchical model as usual, pointing
   ``--thermo_data`` at the HDF5 file.

The HDF5 file is a permanent, reusable artifact.  You only need to regenerate
it if you change the input PDB files or the LigandMPNN model weights.

Installation
------------

Follow the `LigandMPNN installation instructions
<https://github.com/dauparas/LigandMPNN>`_ to create a dedicated conda
environment (LigandMPNN requires Python 3.11 and specific PyTorch pinning).
**Do not install tfscreen** in that environment — the two packages have
incompatible NumPy and JAX dependencies.

The feature generator additionally requires ``h5py``, ``PyYAML``, and
``scipy`` on top of what LigandMPNN itself needs. Copy or clone the tfscreen
repository and run the script directly::

    conda activate ligandmpnn_env
    python /path/to/tfscreen/scripts/generate_struct_ensemble.py ...

Preparing Inputs
----------------

structures YAML
^^^^^^^^^^^^^^^

Create a YAML file with a top-level ``structures:`` mapping from short
structure names to PDB file paths, plus ``n_chains_bearing_mut`` (the number
of chains that carry each mutation: ``1`` for a monomer, ``2`` for a
homodimer). The structure names must match what the downstream theta
component expects. For the four-state (``O2_C4_*``) topologies the required
keys are ``H``, ``HD``, ``L``, and ``LE2``, representing the four
thermodynamic states:

.. code-block:: yaml

    n_chains_bearing_mut: 2
    structures:
      H:   /path/to/H_apo.pdb
      HD:  /path/to/HD_dna_bound.pdb
      L:   /path/to/L_allosteric.pdb
      LE2: /path/to/LE2_iptg_bound.pdb

Each PDB must contain the full protein chain(s) in the conformation
appropriate for that state.  LigandMPNN scores every amino acid position
in the PDB, so make sure the chain contains only the residues you intend
to score. All chains across all PDB files must share the same residue
numbering scheme (the script does not verify this).

PDB residue numbering
^^^^^^^^^^^^^^^^^^^^^

Mutation labels used by the theta component follow the convention
``{wt_aa}{PDB_resnum}{mut_aa}`` (e.g. ``A42G``).  The residue numbers must
match the ``ATOM`` record numbers in the PDB files — not a zero-based index.
Verify that your PDB files use consistent numbering across all structures.

Running the Feature Generator
------------------------------

Activate the LigandMPNN environment, then run::

    python /path/to/tfscreen/scripts/generate_struct_ensemble.py structures.yaml \
        --out ensemble.h5 \
        --ligandmpnn_dir /path/to/LigandMPNN \
        [--model_type ligand_mpnn] \
        [--checkpoint /path/to/weights.pt] \
        [--num_batches 10] \
        [--seed 42]

**Required arguments**

``structures.yaml``
    Path to the YAML file mapping structure names to PDB paths, plus
    ``n_chains_bearing_mut`` (see above).

``--out ensemble.h5``
    Output HDF5 file.

``--ligandmpnn_dir``
    Path to the root of the LigandMPNN repository (the directory that
    contains ``score.py``).

**Optional arguments**

``--model_type``
    LigandMPNN model variant.  Default: ``ligand_mpnn``.  Choices:
    ``ligand_mpnn``, ``protein_mpnn``, ``per_residue_label_membrane_mpnn``,
    ``global_label_membrane_mpnn``, ``soluble_mpnn``.

``--checkpoint``
    Path to a custom model-weights ``.pt`` file.  If omitted, LigandMPNN
    uses its built-in default weights for the selected model type.

``--num_batches``
    Number of random decoding-order batches to average.  More batches reduce
    variance in the log-probability estimates.  Default: ``10``; at least
    ``10`` is recommended.

``--seed``
    Random seed passed to LigandMPNN.  Default: ``42``.

A companion provenance YAML (recording the input structures, LigandMPNN
settings, and generation timestamp) is written alongside the HDF5 file.

HDF5 file contents
^^^^^^^^^^^^^^^^^^^

``structure_names``
    Dataset of structure names, in the order they appear in the input YAML.

For each structure name ``{name}`` the file contains a group with:

``{name}/logP``
    ``float32`` array of shape ``(L, 20)``: mean log P(AA | structure,
    context), averaged across chains and over all decoding-order batches.
    Columns follow the LigandMPNN / ProteinMPNN alphabet
    ``ACDEFGHIKLMNPQRSTVWY`` (indices 0–19).

``{name}/residue_nums``
    ``int32`` array of shape ``(L,)``: PDB residue numbers for each row
    (sorted, unique), used to look up positions by mutation label.

``{name}/dist_matrix``
    ``float32`` array of shape ``(L, L)``: minimum Cα–Cα distance in Å
    across all chain-pair combinations, indexed the same way as
    ``residue_nums``. Used to build distance-dependent contact features for
    pairwise epistasis terms.

``{name}/n_chains_bearing_mut``
    ``int32`` scalar: the ``n_chains_bearing_mut`` value from the input YAML.

Using the HDF5 File in a Run Config
------------------------------------

Pass the HDF5 file to ``tfs-configure-model`` via ``--thermo_data`` when
selecting a ``PnnC`` theta component:

.. code-block:: bash

    tfs-configure-model binding.csv --growth_df growth.csv \
        --theta_model thermo.O2_C4_K3_U0_a.PnnC \
        --thermo_data ensemble.h5

The path is saved into ``tfs_configure_config.yaml`` as the top-level
``thermo_data`` key and read by ``tfscreen.tfmodel.generative.components.theta.thermo.io.load_struct_ensemble``
at model-build time; you do not need to edit the config by hand. See
``configure_model_cli.py``'s ``thermo_data`` docstring for the full set of
registry keys and their structure-name requirements (``PddG`` models instead
take a hand-supplied ΔΔG CSV via the same flag — see
``load_ddG_prior_csv``).
