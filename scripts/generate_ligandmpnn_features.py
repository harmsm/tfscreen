#!/usr/bin/env python3
"""
Generate LigandMPNN per-position log-probability features for tfscreen.

Reads a YAML file mapping structure names to PDB paths, runs LigandMPNN
single_aa_score mode on each structure, and writes a single NPZ file
containing per-structure arrays that can be consumed by the
``lac_dimer_nn_mut`` theta component.

NPZ contents (one pair per structure name)
------------------------------------------
``{name}``              — float32 (L, 20): mean log P(AA | structure, context)
                          averaged over ``--num_batches`` random decoding orders.
                          Columns follow the LigandMPNN / ProteinMPNN alphabet:
                          ACDEFGHIKLMNPQRSTVWY  (index 0–19).
``{name}_residue_nums`` — int32 (L,): PDB residue numbers for each row.

Usage
-----
python generate_ligandmpnn_features.py structures.yaml \\
    --out features.npz \\
    --ligandmpnn_dir /path/to/LigandMPNN \\
    [--model_type ligand_mpnn] \\
    [--checkpoint /path/to/weights.pt] \\
    [--num_batches 10] \\
    [--seed 42]

structures.yaml format
----------------------
H:   /path/to/H_apo.pdb
HD:  /path/to/HD_dna_bound.pdb
L:   /path/to/L_allosteric.pdb
LE2: /path/to/LE2_iptg_bound.pdb

Any set of structure names is accepted; the downstream tfscreen component
(e.g. lac_dimer_nn_mut) declares which keys it requires via STRUCTURE_KEYS.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import yaml

# LigandMPNN standard amino-acid alphabet (20 AA; index 20 = X excluded here).
_ALPHABET_20 = 'ACDEFGHIKLMNPQRSTVWY'

# Matches PDB residue identifiers like "A42", "B100", "A42A" (insertion code).
_RESNAME_RE = re.compile(r'^[A-Za-z](-?\d+)')


def _parse_resnum(res_name):
    """Return the integer residue number from a LigandMPNN residue identifier."""
    m = _RESNAME_RE.match(res_name)
    if m is None:
        raise ValueError(f"Cannot parse residue name: {res_name!r}")
    return int(m.group(1))


def _score_structure(score_py, pdb_path, out_dir,
                     model_type, checkpoint, num_batches, seed):
    """
    Run LigandMPNN score.py on one PDB in single_aa_score mode.

    Returns
    -------
    logP : np.ndarray, shape (L, 20)
        Mean log-probability of each amino acid at each residue position,
        averaged over ``num_batches`` random decoding orders.
    res_nums : np.ndarray, shape (L,), dtype int32
        PDB residue number for each row of logP.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load LigandMPNN output (.pt files). "
            "Install it with: pip install torch"
        ) from exc

    cmd = [
        sys.executable, score_py,
        '--single_aa_score', '1',
        '--model_type', model_type,
        '--pdb_path', pdb_path,
        '--out_folder', out_dir,
        '--use_sequence', '1',
        '--number_of_batches', str(num_batches),
        '--batch_size', '1',
        '--seed', str(seed),
    ]
    if checkpoint:
        flag = ('--checkpoint_ligand_mpnn'
                if model_type == 'ligand_mpnn'
                else '--checkpoint_protein_mpnn')
        cmd += [flag, checkpoint]

    # Run from the LigandMPNN directory so relative model-param paths resolve.
    ligandmpnn_dir = os.path.dirname(score_py)
    subprocess.run(cmd, check=True, cwd=ligandmpnn_dir)

    # Locate the output .pt file.  LigandMPNN writes to {out_folder}/score/.
    pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
    pt_path = os.path.join(out_dir, 'score', f"{pdb_stem}.pt")
    if not os.path.isfile(pt_path):
        # Fallback: some versions write directly to out_folder
        pt_path = os.path.join(out_dir, f"{pdb_stem}.pt")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(
            f"LigandMPNN output not found at {pt_path}. "
            f"Check that score.py ran successfully and wrote to {out_dir}."
        )

    output = torch.load(pt_path, weights_only=False)

    # residue_names: {int_index: str_like_"A42"}
    residue_names = output['residue_names']

    # mean_of_probs: {res_name_str: {aa_letter: float}}
    # Built by score.py from the mean across all decoding-order batches.
    mean_of_probs = output['mean_of_probs']

    L = len(residue_names)
    logP     = np.zeros((L, 20), dtype=np.float32)
    res_nums = np.zeros(L,       dtype=np.int32)

    for idx in range(L):
        res_name = residue_names[idx]
        res_nums[idx] = _parse_resnum(res_name)

        aa_prob_dict = mean_of_probs[res_name]  # {aa_letter: float or tensor}
        for aa_col, aa in enumerate(_ALPHABET_20):
            prob = aa_prob_dict.get(aa, 0.0)
            if hasattr(prob, 'item'):   # unwrap torch scalar
                prob = prob.item()
            logP[idx, aa_col] = np.log(max(float(prob), 1e-10))

    return logP, res_nums


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'structures_yaml',
        help='YAML file mapping structure names to PDB paths',
    )
    parser.add_argument(
        '--out', required=True,
        help='Output NPZ file path',
    )
    parser.add_argument(
        '--ligandmpnn_dir', required=True,
        help='Path to the LigandMPNN repository (must contain score.py)',
    )
    parser.add_argument(
        '--model_type', default='ligand_mpnn',
        choices=[
            'ligand_mpnn', 'protein_mpnn',
            'per_residue_label_membrane_mpnn',
            'global_label_membrane_mpnn', 'soluble_mpnn',
        ],
        help='LigandMPNN model type (default: ligand_mpnn)',
    )
    parser.add_argument(
        '--checkpoint', default=None,
        help=(
            'Path to model weights (.pt). '
            'If omitted, LigandMPNN uses its built-in default for the model type.'
        ),
    )
    parser.add_argument(
        '--num_batches', type=int, default=10,
        help=(
            'Number of random decoding-order batches to average '
            '(≥10 recommended; default: 10)'
        ),
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed passed to LigandMPNN (default: 42)',
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Validate inputs
    if not os.path.isfile(args.structures_yaml):
        parser.error(f"structures_yaml file not found: {args.structures_yaml!r}")
    with open(args.structures_yaml) as fh:
        structures = yaml.safe_load(fh)
    if not isinstance(structures, dict) or not structures:
        parser.error(
            "structures_yaml must be a non-empty YAML mapping of "
            "structure_name: pdb_path pairs."
        )

    score_py = os.path.join(args.ligandmpnn_dir, 'score.py')
    if not os.path.isfile(score_py):
        parser.error(
            f"score.py not found in --ligandmpnn_dir={args.ligandmpnn_dir!r}. "
            f"Make sure the path points to the LigandMPNN repository root."
        )

    for name, pdb_path in structures.items():
        if not os.path.isfile(pdb_path):
            parser.error(f"PDB file for structure '{name}' not found: {pdb_path}")

    # add fully resolved paths to structures
    for s in structures:
        structures[s] = os.path.abspath(structures[s])

    # -------------------------------------------------------------------------
    # Score each structure
    arrays = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, pdb_path in structures.items():
            print(f"[{name}] scoring {pdb_path} ...", flush=True)
            logP, res_nums = _score_structure(
                score_py, pdb_path, tmp_dir,
                model_type=args.model_type,
                checkpoint=args.checkpoint,
                num_batches=args.num_batches,
                seed=args.seed,
            )
            arrays[name]                       = logP
            arrays[f"{name}_residue_nums"]     = res_nums
            print(f"[{name}] → (L={logP.shape[0]}, 20 AA) done", flush=True)

    # -------------------------------------------------------------------------
    # Save NPZ
    np.savez(args.out, **arrays)
    print(f"\nWrote {len(structures)} structure(s) to {args.out}")
    print("Keys:", ", ".join(sorted(arrays.keys())))


if __name__ == '__main__':
    main()
