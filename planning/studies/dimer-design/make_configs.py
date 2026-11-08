"""
Write the dimer-design simulate configs (plan step 4, 4c-4d), one run
directory per simulation.

The question is whether the NSF proposal's background design
(~/work/Documents/writing/grants/2026-11-20_nsf-grant/analyses/
2026-11-04_results-narrative.md) resolves the per-mutation constants
when the truth is the full thermo.dimer model, as its Fisher calculation
on a reduced ensemble predicts (0.09-0.19 kcal/mol for five separated
double backgrounds).

Base: inputs/base_smoke.yaml (the A-series smoke library, subsampled
smaller here) with thermo.dimer.PddG truth, monomers and the unfolded
monomer on and the unfolded dimer off (the narrative's ensemble: folding
per monomer, then dimerization; with U2 on as well, a dimer state's
ΔΔG_H also drives Δln K_u and unfolds the folding background L114T/I289D
completely, θ ≈ 0 at every TF level), from the 2026-11-03 state table (kcal/mol at 310.15 K) plus
DDG_TRUTH_SD noise. wt is held at the model defaults except the monomer
terms, set from the narrative's assumptions: folding -4 kcal/mol per
monomer (ln_K_m_u) and dimerization DIMER kcal/mol (ln_K_d_h). The
narrative works at 0.1 uM repressor and our wt is calibrated at 10 uM;
free monomer goes as sqrt(Kd / [TF]), so 100-fold more TF is about 2.8
kcal/mol of dimerization, and the -12/-15/-18 dial brackets its case.

Arms (one read budget, TOTAL_TUBES_BUDGET tubes' worth of A2 reads):

    W   reference only, A2's 118-tube design at TF mid
    N   W plus the narrative's five separated doubles (2026-11-04_
        separated-backgrounds), each a 20-tube block at its TF level
    R   W plus RANDOM_N random single-mutant backgrounds at TF mid, the
        count control ("composition beats count")

    G   W plus the ten greedy backgrounds of background_count.py (greedy
        on predicted θ error in held-out contexts), each a 20-tube block

Runs: {W,N,R}_c1 and {W,N,R}_c03 (table slope ddG_truth_scale 1 and 0.3,
dimerization -15), N_c1_d12 and N_c1_d18 (dimerization -12 and -18): the
pilots, seed 1.

The error panel (2026-11-07), dimerization -15, seeds 1-3 (seed 1 is the
pilot's run name with no suffix, seeds 2 and 3 add _s2/_s3):
    {W,N,G}_c1     random table error only (DDG_TRUTH_SD), c = 1
    {W,N,G}_c03    plus systematic compression, c = 0.3
    {W,N,G}_ep     c = 1 plus random pair epistasis in the truth: the pair
                   horseshoe's global scale prior at EPI_TAU_SCALE instead of
                   1e-9 (independent across terms; no distances in PddG)
Model mis-specification is a fit arm on the c1 runs (run_fit.sh, *_m0),
not a simulation.

Run from this directory:  ~/miniconda3/bin/python make_configs.py
"""

import copy
import os

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = "2026-11-03_laci-state-ddg.csv"

KT_PER_KCAL = 1.0 / (1.987204259e-3 * 310.15)     # 1.6225
DDG_TRUTH_SD = 0.5          # kcal/mol, as in the A series
WT_PIN = 0.001              # wt prior SD, so the truth's wt is set here
FOLD_KCAL = -4.0            # U -> H per monomer (narrative)
TF_MID_UM = 10.0            # held; lo and hi learned

# Each background is its own strain and so its own library per marker,
# with the reference's design (singles and doubles; a background cannot
# drop a tile combo) and no spikes. The same subsample keeps the same
# design members in every background.
SUBSAMPLE = {"single-1": 50, "single-2": 50, "double-1-2": 25}

# The narrative's final design (separated-backgrounds, panel B):
# background mutations and the TF level each is measured at.
NARRATIVE = [
    (("L114T", "I289D"), "hi"),     # folding +7.94
    (("R51P", "C281D"), "hi"),      # interface +7.36
    (("S61P", "D149M"), "mid"),     # conformational -3.42, effector +2.50
    (("S97A", "G187E"), "lo"),      # conformational +2.82, effector +2.50
    (("R51S", "D149M"), "mid"),     # conformational -3.35
]
CODON = {"P": "ccg", "A": "gcg", "S": "agc"}

RANDOM_N = 15
RANDOM_SEED = 20261104
RANDOM_MAX_KCAL = 2.0       # "readable" proxy: every state within 2 kcal/mol

BLOCK_TUBES = 20
TOTAL_TUBES_BUDGET = 118 + 5 * BLOCK_TUBES      # the N arm, at A2's reads
A2_TUBES = 118

CONTEXTS = {
    "units": "kT",
    "operators": {"O1": {"reference": True}},
    "tf_levels": {"mid": {"conc_uM": TF_MID_UM, "ln_sd": 0.0},
                  "lo": {"conc_uM": TF_MID_UM / 3.0, "ln_sd": 0.5},
                  "hi": {"conc_uM": TF_MID_UM * 3.0, "ln_sd": 0.5}},
}
BINDING = {"operator": "O1", "tf_total_uM": 4.0}   # 2 uM dimer in vitro

WT_SITES = ["ln_K_h_l", "ln_K_h_o_eff", "ln_K_l_o_eff", "ln_K_u",
            "ln_K_h_oii", "ln_K_l_oii", "ln_K_h_e", "ln_K_l_e",
            "ln_K_d_h", "ln_K_m_h_l", "ln_K_m_h_o", "ln_K_m_l_o", "ln_K_m_u"]

EPI_TAU_SCALE = 0.3         # truth pair horseshoe global scale (ep runs)

# background_count.py, greedy on held_out (2026-11-07, c = 1, sigma 0.1)
GREEDY = [
    (("V9R", "K290D"), "hi"), (("Q117L", "F293A"), "hi"), (("Y17P",), "hi"),
    (("Q117T",), "lo"), (("A116D", "K314P"), "hi"), (("V324R", "D129V"), "lo"),
    (("P3F",), "lo"), (("V99G",), "lo"), (("T5I",), "lo"),
    (("V4L", "P3R"), "hi"),
]

RUNS = {
    "W_c1": ("W", 1.0, -15.0), "N_c1": ("N", 1.0, -15.0),
    "R_c1": ("R", 1.0, -15.0),
    "W_c03": ("W", 0.3, -15.0), "N_c03": ("N", 0.3, -15.0),
    "R_c03": ("R", 0.3, -15.0),
    "N_c1_d12": ("N", 1.0, -12.0), "N_c1_d18": ("N", 1.0, -18.0),
}
# The error panel: (arm, slope, dimerization, epistasis, seed).
PANEL = {}
for _arm in ("W", "N", "G"):
    for _cond in ("c1", "c03", "ep"):
        for _seed in (1, 2, 3):
            _run = f"{_arm}_{_cond}" + (f"_s{_seed}" if _seed > 1 else "")
            PANEL[_run] = (_arm, 0.3 if _cond == "c03" else 1.0, -15.0,
                           _cond == "ep", _seed)


def amplicon(cf):
    first = int(cf["first_amplicon_residue"])
    return first, first + len(cf["wt_seq"]) // 3 - 1


def background_spec(muts, cf):
    """A backgrounds-block entry: in-amplicon mutations edit wt_seq, the
    rest are outside_mutations."""
    first, last = amplicon(cf)
    seq = list(cf["wt_seq"])
    outside, edited = [], False
    for m in muts:
        res = int(m[1:-1])
        if first <= res <= last:
            i = 3 * (res - first)
            seq[i:i + 3] = CODON[m[-1]]
            edited = True
        else:
            outside.append(m)
    spec = {}
    if edited:
        spec["wt_seq"] = "".join(seq)
    if outside:
        spec["outside_mutations"] = outside
    return spec


def random_backgrounds(cf):
    """RANDOM_N single mutations outside the amplicon, one per residue,
    every state within RANDOM_MAX_KCAL (a proxy for 'leaves a readable
    curve'), drawn with RANDOM_SEED."""
    table = pd.read_csv(os.path.join(HERE, "inputs", TABLE))
    first, last = amplicon(cf)
    res = table["mut"].str[1:-1].astype(int)
    states = table.drop(columns="mut")
    ok = (states.abs().max(axis=1) <= RANDOM_MAX_KCAL) & \
        ((res < first) | (res > last)) & (table["mut"].str[-1] != "*")
    pool = table.loc[ok, "mut"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    out, used = [], set()
    for m in rng.permutation(pool):
        r = int(m[1:-1])
        if r in used:
            continue
        used.add(r)
        out.append(str(m))
        if len(out) == RANDOM_N:
            break
    return out


def arm_backgrounds(arm, cf):
    """[(name, spec, tf_level)] for an arm's non-reference backgrounds."""
    if arm == "W":
        return []
    if arm in ("N", "G"):
        return [("bg" + "_".join(muts), background_spec(muts, cf), lvl)
                for muts, lvl in (NARRATIVE if arm == "N" else GREEDY)]
    return [(f"bg{m}", background_spec((m,), cf), "mid")
            for m in random_backgrounds(cf)]


def arm_design(design, bgs):
    """The reference design at TF mid plus a 20-tube block per background:
    replicate 1, the middle time of each condition and concentration."""
    blocks = [design.assign(background="reference", operator="O1",
                            tf_level="mid")]
    rep1 = design[design["replicate"] == 1]
    middle = (rep1.sort_values("t_sel")
              .groupby(["library", "condition_sel", "titrant_conc"])
              .nth(1))
    assert len(middle) == BLOCK_TUBES
    for name, _, lvl in bgs:
        block = middle.assign(background=name, operator="O1", tf_level=lvl)
        block["sample"] = block["sample"] + f"_{name}_{lvl}"
        block["library"] = block["library"] + f"_{name}"
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def run_config(base, arm, scale, dimer_kcal, design_file, bgs, design,
               epistasis=False, seed=None):
    cf = copy.deepcopy(base)
    cf["design"] = f"../../inputs/{design_file}"
    cf["theta_component"] = "thermo.dimer.PddG"
    cf["thermo_data"] = f"../../inputs/{TABLE}"
    cf["thermo_options"] = {"ddG_units": "kcal/mol", "temperature_K": 310.15}
    cf["ddG_truth_sd"] = DDG_TRUTH_SD
    if scale != 1.0:
        cf["ddG_truth_scale"] = scale
    cf["contexts"] = f"../../inputs/contexts_{arm}.yaml"
    cf["binding_data"].update(BINDING)
    cf["od600"]["calibration"] = "../../inputs/od600_calibration.yaml"
    cf["library_subsample"] = dict(SUBSAMPLE)
    if bgs:
        cf["backgrounds"] = {name: spec for name, spec, _ in bgs}
    priors = {f"theta_{s}_wt_scale": WT_PIN for s in WT_SITES}
    priors["theta_ln_K_d_h_wt_loc"] = dimer_kcal * -KT_PER_KCAL
    priors["theta_ln_K_m_u_wt_loc"] = FOLD_KCAL * KT_PER_KCAL
    priors["theta_epi_tau_scale"] = EPI_TAU_SCALE if epistasis else 1.0e-9
    priors["theta_monomers"] = True
    priors["theta_unfolded_monomer"] = True
    priors["theta_unfolded_dimer"] = False
    cf["theta_priors"] = priors
    if seed is not None and seed != 1:
        cf["seed"] = int(seed)
    libraries = sorted(set(design["library"]))
    for key in ("cfu0", "unassigned_read_fraction"):
        cf[key] = {lib: base[key][lib.split("_")[0]] for lib in libraries}
    # one read budget for every arm
    cf["total_num_reads"] = int(round(base["total_num_reads"]
                                      * TOTAL_TUBES_BUDGET / A2_TUBES))
    return cf


def main():
    with open(os.path.join(HERE, "inputs", "base_smoke.yaml")) as f:
        base = yaml.safe_load(f)
    base.pop("theta_priors", None)
    design = pd.read_csv(os.path.join(HERE, "inputs", "design.csv"))
    assert len(design) == A2_TUBES
    arms = {}
    for arm in ("W", "N", "R", "G"):
        bgs = arm_backgrounds(arm, base)
        d = arm_design(design, bgs)
        name = f"design_{arm}.csv"
        d.to_csv(os.path.join(HERE, "inputs", name), index=False)
        arms[arm] = (bgs, d, name)
        # check_contexts refuses a level no data row has, so each arm's
        # contexts file names only the TF levels its design uses
        used = set(d["tf_level"])
        ctx = dict(CONTEXTS, tf_levels={k: v for k, v in
                                        CONTEXTS["tf_levels"].items()
                                        if k in used})
        with open(os.path.join(HERE, "inputs", f"contexts_{arm}.yaml"),
                  "w") as f:
            f.write(f"# dimer-design contexts, arm {arm}: written by "
                    "make_configs.py\n")
            yaml.safe_dump(ctx, f, sort_keys=False)
        print(f"arm {arm}: {len(d)} tubes, {len(bgs)} backgrounds"
              + (f" ({', '.join(b for b, _, _ in bgs)})" if bgs else ""))

    runs = {r: (a, s, k, False, None) for r, (a, s, k) in RUNS.items()}
    runs.update(PANEL)
    for run, (arm, scale, dimer_kcal, epi, seed) in runs.items():
        bgs, d, name = arms[arm]
        cf = run_config(base, arm, scale, dimer_kcal, name, bgs, d, epi, seed)
        out = os.path.join(HERE, "runs", run)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "tfs_sim_config.yaml"), "w") as f:
            f.write(f"# dimer-design run {run}: arm {arm}, table slope "
                    f"{scale}, dimerization {dimer_kcal} kcal/mol"
                    + (", pair epistasis" if epi else "")
                    + (f", seed {seed}" if seed and seed != 1 else "")
                    + "; written by make_configs.py\n")
            yaml.safe_dump(cf, f, sort_keys=False)
        print(f"wrote runs/{run}/tfs_sim_config.yaml")


if __name__ == "__main__":
    main()
