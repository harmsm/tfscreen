#!/usr/bin/env python3
"""
Generate structural ensemble features for tfscreen struct theta models.

For each conformational state listed in the input YAML this script:

  1. Runs LigandMPNN (single-AA scoring mode) to get per-residue
     log-probabilities over the 20-AA alphabet.
  2. Aggregates logP rows across chains by averaging — appropriate for
     symmetric homodimers where both chains see the same structural context.
  3. Parses Cα coordinates from the PDB and computes the minimum inter-
     residue Cα-Cα distance across *all* chain-pair combinations, so that
     inter-chain contacts in a dimer are captured at their true distance.
  4. Assembles a single HDF5 file + companion provenance YAML.

HDF5 schema
-----------
structure_names              — str dataset : names of all structures in this file
{name}/logP                  — float32 (L, 20) : logP averaged over chains;
                               rows indexed by unique PDB residue number
{name}/residue_nums          — int32   (L,)    : unique residue numbers (sorted)
{name}/dist_matrix           — float32 (L, L)  : min Cα-Cα distance in Å across
                               all chain-pair combinations
{name}/n_chains_bearing_mut  — int32 scalar    : chains that carry a mutation

Input YAML format
-----------------
n_chains_bearing_mut: 2        # chains that carry each mutation (1=monomer, 2=homodimer)
structures:
  H:   /path/to/H_apo.pdb
  HD:  /path/to/HD_bound.pdb
  L:   /path/to/L_allosteric.pdb
  LE2: /path/to/LE2_iptg.pdb

Usage
-----
python generate_struct_ensemble.py structures.yaml \\
    --out ensemble.h5 \\
    --ligandmpnn_dir /path/to/LigandMPNN \\
    [--model_type ligand_mpnn] \\
    [--checkpoint /path/to/weights.pt] \\
    [--num_batches 10] \\
    [--seed 42]


Assumptions
-----------
All chains in every PDB file must share the same residue numbering
scheme: residue 42 in chain A and residue 42 in chain B must correspond
to the same sequence position.  This script does **not** verify that
assumption.  If chains use different numbering (e.g. chain B starts at
residue 1 while chain A starts at residue 101), the logP rows from each
chain will be averaged under the wrong residue numbers and the distance
matrix will be silently incorrect.  For standard symmetric homodimer PDB
files this condition is always met.
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import yaml
from scipy.spatial.distance import cdist

# LigandMPNN / ProteinMPNN standard 20-AA alphabet.
_ALPHABET_20 = 'ACDEFGHIKLMNPQRSTVWY'

# Matches LigandMPNN residue identifiers like "A42", "B100", "A42A".
_RESNAME_RE = re.compile(r'^[A-Za-z](-?\d+)')


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def _parse_pdb_ca(pdb_path):
    """
    Extract Cα coordinates for every chain in a PDB file.

    Insertion codes are ignored (residue 42 and 42A are both mapped to 42;
    the first ATOM record wins). HETATM records are skipped.

    Returns
    -------
    dict: chain_id (str) -> dict: resnum (int) -> coords np.ndarray (3,) float32
    """
    chains = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            if line[12:16].strip() != 'CA':
                continue
            chain_id = line[21]
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if chain_id not in chains:
                chains[chain_id] = {}
            if resnum not in chains[chain_id]:   # keep first model / alt conf
                chains[chain_id][resnum] = np.array([x, y, z], dtype=np.float32)
    if not chains:
        raise ValueError(f"No Cα atoms found in {pdb_path!r}")
    return chains


def _build_min_dist_matrix(ca_chains, unique_resnums):
    """
    Build a (n, n) minimum Cα-Cα distance matrix over all chain pairs.

    For each pair of residue positions (i, j), the stored value is the
    minimum Euclidean distance across *all* combinations of chains — so
    for a homodimer both intra- and inter-chain distances are considered
    and the smaller one is kept.

    Residues in unique_resnums that are absent from the PDB get distance
    999 Å (they will not form contact pairs after any realistic cutoff).

    Parameters
    ----------
    ca_chains : dict  chain_id -> {resnum: coords}
    unique_resnums : list of int  (sorted, no duplicates)

    Returns
    -------
    np.ndarray, shape (n, n), dtype float32
    """
    n = len(unique_resnums)
    resnum_to_idx = {r: i for i, r in enumerate(unique_resnums)}
    min_dist = np.full((n, n), 999.0, dtype=np.float32)

    # Build one (k, 3) coordinate block per chain, aligned to unique_resnums.
    chain_blocks = []
    for chain_id, coord_map in ca_chains.items():
        idx_list = []
        xyz_list = []
        for r in unique_resnums:
            if r in coord_map:
                idx_list.append(resnum_to_idx[r])
                xyz_list.append(coord_map[r])
        if idx_list:
            chain_blocks.append((
                np.array(idx_list, dtype=np.intp),
                np.array(xyz_list, dtype=np.float32),
            ))

    for idx_a, xyz_a in chain_blocks:
        for idx_b, xyz_b in chain_blocks:
            dists = cdist(xyz_a, xyz_b).astype(np.float32)   # (n_a, n_b)
            current = min_dist[np.ix_(idx_a, idx_b)]
            min_dist[np.ix_(idx_a, idx_b)] = np.minimum(current, dists)

    return min_dist


# ---------------------------------------------------------------------------
# LigandMPNN interface
# ---------------------------------------------------------------------------

def _parse_resnum(res_name):
    """Return integer residue number from a LigandMPNN identifier like 'A42'."""
    m = _RESNAME_RE.match(res_name)
    if m is None:
        raise ValueError(f"Cannot parse residue name: {res_name!r}")
    return int(m.group(1))


def _score_structure(score_py, pdb_path, out_dir,
                     model_type, checkpoint, num_batches, seed):
    """
    Run LigandMPNN score.py in single_aa_score mode on one PDB.

    Returns
    -------
    logP_raw  : float32 (L_raw, 20)  all chains, duplicate resnums possible
    res_nums  : int32   (L_raw,)
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load LigandMPNN output (.pt files). "
            "Install with: pip install torch"
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

    ligandmpnn_dir = os.path.dirname(score_py)
    subprocess.run(cmd, check=True, cwd=ligandmpnn_dir)

    pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
    pt_path = os.path.join(out_dir, 'score', f"{pdb_stem}.pt")
    if not os.path.isfile(pt_path):
        pt_path = os.path.join(out_dir, f"{pdb_stem}.pt")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(
            f"LigandMPNN output not found at {pt_path}. "
            f"Check that score.py ran successfully and wrote to {out_dir}."
        )

    output = torch.load(pt_path, weights_only=False)
    residue_names = output['residue_names']
    mean_of_probs = output['mean_of_probs']

    L = len(residue_names)
    logP_raw = np.zeros((L, 20), dtype=np.float32)
    res_nums = np.zeros(L, dtype=np.int32)

    for idx in range(L):
        res_name = residue_names[idx]
        res_nums[idx] = _parse_resnum(res_name)
        aa_prob_dict = mean_of_probs[res_name]
        for aa_col, aa in enumerate(_ALPHABET_20):
            prob = aa_prob_dict.get(aa, 0.0)
            if hasattr(prob, 'item'):
                prob = prob.item()
            logP_raw[idx, aa_col] = np.log(max(float(prob), 1e-10))

    return logP_raw, res_nums


def _aggregate_logP(logP_raw, res_nums_raw):
    """
    Average per-chain logP rows into one row per unique residue number.

    LigandMPNN returns one row per residue across all chains, so residue
    numbers are duplicated in multi-chain structures (e.g. "A42" and "B42"
    both map to residue 42). This averages log-probabilities across chains
    (geometric mean of raw probabilities), which is appropriate for symmetric
    homodimers where both chains occupy the same structural environment.

    Returns
    -------
    logP_agg      : float32 (n_unique, 20)
    unique_resnums : int32  (n_unique,)  sorted
    """
    unique_list = sorted(set(res_nums_raw.tolist()))
    n_unique = len(unique_list)
    resnum_to_idx = {r: i for i, r in enumerate(unique_list)}

    logP_sum = np.zeros((n_unique, 20), dtype=np.float64)
    counts   = np.zeros(n_unique, dtype=np.int32)

    for raw_i, rnum in enumerate(res_nums_raw):
        if rnum in resnum_to_idx:
            idx = resnum_to_idx[rnum]
            logP_sum[idx] += logP_raw[raw_i]
            counts[idx]   += 1

    logP_agg = (logP_sum / np.maximum(counts[:, None], 1)).astype(np.float32)
    return logP_agg, np.array(unique_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# Config and provenance
# ---------------------------------------------------------------------------

def _load_config(yaml_path):
    """
    Load and validate the structures YAML config.

    Returns
    -------
    structures           : dict  name -> absolute pdb path (str)
    n_chains_bearing_mut : int
    """
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping.")

    if 'structures' not in cfg:
        raise ValueError(
            "Config YAML must contain a 'structures:' key.\n"
            "Expected format:\n\n"
            "  n_chains_bearing_mut: 2\n"
            "  structures:\n"
            "    H:   /path/to/H.pdb\n"
            "    HD:  /path/to/HD.pdb\n"
        )

    n_chains = int(cfg.get('n_chains_bearing_mut', 1))
    if n_chains < 1:
        raise ValueError(f"n_chains_bearing_mut must be >= 1, got {n_chains}")

    structures_raw = cfg['structures']
    if not isinstance(structures_raw, dict) or not structures_raw:
        raise ValueError("'structures' must be a non-empty name -> pdb_path mapping.")

    structures = {}
    for name, path in structures_raw.items():
        abs_path = os.path.abspath(str(path))
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(
                f"PDB for structure '{name}' not found: {path!r}"
            )
        structures[name] = abs_path

    return structures, n_chains


def _write_provenance_yaml(out_npz, config_path, structures, n_chains,
                           model_type, checkpoint, num_batches, seed):
    """Write companion provenance YAML alongside the NPZ."""
    prov = {
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'script': os.path.basename(__file__),
        'config': os.path.abspath(config_path),
        'n_chains_bearing_mut': n_chains,
        'ligandmpnn': {
            'model_type': model_type,
            'checkpoint': checkpoint,
            'num_batches': num_batches,
            'seed': seed,
        },
        'structures': {name: path for name, path in structures.items()},
    }
    yaml_out = os.path.splitext(out_npz)[0] + '.yaml'
    with open(yaml_out, 'w') as fh:
        yaml.dump(prov, fh, default_flow_style=False, sort_keys=False)
    return yaml_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'structures_yaml',
        help='YAML config (structures + n_chains_bearing_mut)',
    )
    parser.add_argument(
        '--out', required=True,
        help='Output HDF5 file path (e.g. ensemble.h5)',
    )
    parser.add_argument(
        '--ligandmpnn_dir', required=True,
        help='Path to the LigandMPNN repository root (must contain score.py)',
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
        help='Path to model weights (.pt); uses LigandMPNN default if omitted',
    )
    parser.add_argument(
        '--num_batches', type=int, default=10,
        help='Decoding-order batches to average (>=10 recommended; default: 10)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed passed to LigandMPNN (default: 42)',
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    if not os.path.isfile(args.structures_yaml):
        parser.error(f"Config YAML not found: {args.structures_yaml!r}")

    structures, n_chains = _load_config(args.structures_yaml)

    score_py = os.path.join(args.ligandmpnn_dir, 'score.py')
    if not os.path.isfile(score_py):
        parser.error(
            f"score.py not found in --ligandmpnn_dir={args.ligandmpnn_dir!r}. "
            f"Ensure this path points to the LigandMPNN repository root."
        )

    # ------------------------------------------------------------------
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to write ensemble files. "
            "Install with: pip install h5py"
        ) from exc

    struct_data = {}   # name -> dict of arrays, filled inside the tmp dir block

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, pdb_path in structures.items():
            print(f"\n[{name}]  {pdb_path}", flush=True)

            # 1. LigandMPNN scoring
            print("  running LigandMPNN ...", flush=True)
            logP_raw, res_nums_raw = _score_structure(
                score_py, pdb_path, tmp_dir,
                model_type=args.model_type,
                checkpoint=args.checkpoint,
                num_batches=args.num_batches,
                seed=args.seed,
            )

            # 2. Aggregate across chains
            logP_agg, unique_resnums = _aggregate_logP(logP_raw, res_nums_raw)
            n_raw    = len(res_nums_raw)
            n_unique = len(unique_resnums)
            n_chains_in_output = n_raw // n_unique if n_unique else 0
            print(f"  logP: {n_raw} rows → {n_unique} unique residues "
                  f"({n_chains_in_output} chain(s) in LigandMPNN output)", flush=True)

            # 3. Cα distance matrix (minimum across all chain pairs)
            print("  computing Cα distance matrix ...", flush=True)
            ca_chains = _parse_pdb_ca(pdb_path)
            dist_matrix = _build_min_dist_matrix(ca_chains, unique_resnums.tolist())
            d_min = dist_matrix[dist_matrix < 999].min() if (dist_matrix < 999).any() else float('nan')
            print(f"  dist matrix: ({n_unique} × {n_unique}),  "
                  f"min distance = {d_min:.2f} Å  "
                  f"({len(ca_chains)} chain(s) in PDB)", flush=True)

            struct_data[name] = {
                'logP':                 logP_agg,
                'residue_nums':         unique_resnums,
                'dist_matrix':          dist_matrix,
                'n_chains_bearing_mut': np.int32(n_chains),
            }

    # ------------------------------------------------------------------
    # Write HDF5: one group per structure, generic keys within each group.
    #
    #   ensemble.h5
    #   ├── structure_names      str dataset
    #   ├── H/
    #   │   ├── logP             float32 (L, 20)
    #   │   ├── residue_nums     int32   (L,)
    #   │   ├── dist_matrix      float32 (L, L)
    #   │   └── n_chains_bearing_mut  int32 scalar
    #   ├── HD/ ...
    with h5py.File(args.out, 'w') as hf:
        names = list(structures.keys())
        hf.create_dataset('structure_names',
                          data=np.array(names, dtype=h5py.string_dtype()))
        for name, arrays in struct_data.items():
            grp = hf.create_group(name)
            grp.create_dataset('logP',                 data=arrays['logP'])
            grp.create_dataset('residue_nums',         data=arrays['residue_nums'])
            grp.create_dataset('dist_matrix',          data=arrays['dist_matrix'])
            grp.create_dataset('n_chains_bearing_mut', data=arrays['n_chains_bearing_mut'])

    print(f"\nWrote HDF5:            {args.out}")

    yaml_out = _write_provenance_yaml(
        args.out, args.structures_yaml, structures, n_chains,
        args.model_type, args.checkpoint, args.num_batches, args.seed,
    )
    print(f"Wrote provenance YAML: {yaml_out}")
    print(f"\nStructures ({len(structures)}): {', '.join(structures.keys())}")
    print(f"n_chains_bearing_mut:  {n_chains}")


if __name__ == '__main__':
    main()
