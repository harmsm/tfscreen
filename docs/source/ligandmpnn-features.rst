===================
LigandMPNN Features
===================

Some tfscreen theta components (e.g. ``lac_dimer_nn_mut``) use per-position
log-probability features from `LigandMPNN
<https://github.com/dauparas/LigandMPNN>`_ as structural inputs. These
features are computed once — outside the main tfscreen environment — and
stored in an NPZ file that is read at analysis time.

.. warning::

   The ``tfs-generate-ligandmpnn-features`` command is the **only** part of
   tfscreen that is guaranteed to work inside the LigandMPNN conda environment.
   LigandMPNN pins specific versions of PyTorch and NumPy that are incompatible
   with the JAX/NumPyro stack that the rest of tfscreen requires.  Do not
   attempt to run the hierarchical model or any other tfscreen analysis
   commands from the LigandMPNN environment.

Overview
--------

The feature generation pipeline is a two-step, two-environment workflow:

1. **LigandMPNN environment** — run ``tfs-generate-ligandmpnn-features`` to
   score each PDB structure and write an NPZ feature file.
2. **tfscreen environment** — run the hierarchical model as usual; point the
   relevant theta component at the NPZ file via the run config YAML.

The NPZ file is a permanent, reusable artifact.  You only need to regenerate
it if you change the input PDB files or the LigandMPNN model weights.

Installation
------------

Follow the `LigandMPNN installation instructions
<https://github.com/dauparas/LigandMPNN>`_ to create a dedicated conda
environment (LigandMPNN requires Python 3.11 and specific PyTorch pinning).
Then install tfscreen **inside that same environment**::

    conda activate ligandmpnn_env
    pip install tfscreen

This makes ``tfs-generate-ligandmpnn-features`` available on your PATH inside
the LigandMPNN environment.  Only that command is supported there; the rest of
the tfscreen CLI requires the normal tfscreen environment.

Preparing Inputs
----------------

structures YAML
^^^^^^^^^^^^^^^

Create a YAML file that maps short structure names to PDB file paths.  The
names must match what the downstream theta component expects.  For
``lac_dimer_nn_mut`` the required keys are ``H``, ``HD``, ``L``, and ``LE2``,
representing the four thermodynamic states:

.. code-block:: yaml

    H:   /path/to/H_apo.pdb
    HD:  /path/to/HD_dna_bound.pdb
    L:   /path/to/L_allosteric.pdb
    LE2: /path/to/LE2_iptg_bound.pdb

Each PDB must contain the full protein chain(s) in the conformation
appropriate for that state.  LigandMPNN scores every amino acid position
in the PDB, so make sure the chain contains only the residues you intend
to score.

PDB residue numbering
^^^^^^^^^^^^^^^^^^^^^

Mutation labels used by the theta component follow the convention
``{wt_aa}{PDB_resnum}{mut_aa}`` (e.g. ``A42G``).  The residue numbers must
match the ``ATOM`` record numbers in the PDB files — not a zero-based index.
Verify that your PDB files use consistent numbering across all four
structures.

Running the Feature Generator
-----------------------------

Activate the LigandMPNN environment, then run::

    tfs-generate-ligandmpnn-features structures.yaml \
        --out features.npz \
        --ligandmpnn_dir /path/to/LigandMPNN \
        [--model_type ligand_mpnn] \
        [--checkpoint /path/to/weights.pt] \
        [--num_batches 10] \
        [--seed 42]

**Required arguments**

``structures.yaml``
    Path to the YAML file mapping structure names to PDB paths (see above).

``--out features.npz``
    Output file.  A single NPZ is written containing one array per structure
    name and one residue-number index array per structure.

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

NPZ file contents
^^^^^^^^^^^^^^^^^

For each structure name ``{name}`` the NPZ contains:

``{name}``
    ``float32`` array of shape ``(L, 20)``: mean log P(AA | structure,
    context) averaged over all decoding-order batches.  Columns follow the
    LigandMPNN / ProteinMPNN alphabet ``ACDEFGHIKLMNPQRSTVWY`` (indices 0–19).

``{name}_residue_nums``
    ``int32`` array of shape ``(L,)``: PDB residue numbers for each row,
    used to look up positions by mutation label.

Using the NPZ in a Run Config
-----------------------------

Point the ``ligandmpnn_features`` field in your run config YAML at the NPZ
file.  The exact field name depends on the theta component; for
``lac_dimer_nn_mut`` it is typically set under the theta component block.
Refer to the component's ``get_hyperparameters()`` for the full list of
required config keys.
