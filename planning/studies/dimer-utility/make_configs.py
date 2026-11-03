"""
Write the stage A simulate configs, one run directory per simulation.

Base: inputs/base_smoke.yaml, the BGMP structural-prior pilot v2 smoke
config (100 + 100 singles, 50 doubles and nine spikes from the 0.5.0
library design, counts-likelihood growth in 118 tubes, spiked and library
binding). We change the theta model and its truth, and point the paths at
inputs/.

    runs/A1/tfs_sim_config.yaml   O2_C12_K5_U1_a.PddG truth: the dimer model
                                  with the second engagement off is this
                                  model, so a dimer.PK fit and an O2 PK fit
                                  of it should agree
    runs/A2/tfs_sim_config.yaml   thermo.dimer.PddG truth, default variant
                                  (unfolded dimer, second engagement)
    runs/A3/tfs_sim_config.yaml   A2 with two operators, two TF levels and
                                  a second background (plan step 3g): A2's
                                  design at (reference, O1, lo) plus five
                                  star arms (inputs/design_A3.csv), the
                                  contexts in inputs/contexts_A3.yaml

Both draw the truth from the AF3 state table (kcal/mol at 310.15 K) plus
ddG_truth_sd per mutation and state, with wt held at the model defaults
and no pair epistasis in the truth.

Run from this directory:  ~/miniconda3/bin/python make_configs.py
"""

import copy
import os

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

DDG_TRUTH_SD = 0.5      # kcal/mol
WT_PIN = 0.001          # wt prior SD, so the truth's wt is the default

COMMON = {
    "design": "../../inputs/design.csv",
    "thermo_data": "../../inputs/2026-10-11_laci-state-ddg.csv",
    "thermo_options": {"ddG_units": "kcal/mol", "temperature_K": 310.15},
    "ddG_truth_sd": DDG_TRUTH_SD,
}

RUNS = {
    "A1": {
        "theta_component": "thermo.O2_C12_K5_U1_a.PddG",
        "wt_sites": ["ln_K_h_l", "ln_K_h_o", "ln_K_h_e", "ln_K_l_o",
                     "ln_K_l_e", "ln_K_u"],
    },
    "A2": {
        "theta_component": "thermo.dimer.PddG",
        "wt_sites": ["ln_K_h_l", "ln_K_h_o_eff", "ln_K_l_o_eff", "ln_K_u",
                     "ln_K_h_oii", "ln_K_l_oii", "ln_K_h_e", "ln_K_l_e"],
    },
}


# A3 (plan step 3g). T276N lies outside the amplicon; in the AF3 table it
# shifts the H/L balance (L +1.34, H +0.18 kcal/mol) and not the effector
# states, the kind of background identifiability step 1 says resolves the
# terms A2's data never saw (l_o, u, l_oii).
A3_BACKGROUND = {"bgT276N": {"outside_mutations": ["T276N"]}}
# The lo level is held at the model's default total TF (no monomers in the
# default variant, so the absolute level is not identifiable); hi is
# learned about 30 uM. O2 binds more weakly than O1 by about 1.5 kT.
A3_CONTEXTS = {
    "units": "kT",
    "operators": {"O1": {"reference": True},
                  "O2": {"shift": 1.5, "shift_sd": 0.5}},
    "tf_levels": {"lo": {"conc_uM": 10.0, "ln_sd": 0.0},
                  "hi": {"conc_uM": 30.0, "ln_sd": 0.5}},
}
# (background, operator, TF level) of each design block. The first is A2's
# whole design; every other is a star arm, each its own strain and so its
# own library per marker.
A3_BASE = ("reference", "O1", "lo")
A3_ARMS = [("bgT276N", "O1", "lo"),
           ("reference", "O2", "lo"), ("bgT276N", "O2", "lo"),
           ("reference", "O1", "hi"), ("bgT276N", "O1", "hi")]
A3_BINDING = {"operator": "O1", "tf_total_uM": 4.0}   # 2 uM dimer in vitro


def _library_name(marker, block):
    if block == A3_BASE:
        return marker
    bg, op, lvl = block
    return f"{marker}_{'ref' if bg == 'reference' else bg}_{op}_{lvl}"


def a3_design(design):
    """A2's design at the base block plus the star arms: replicate 1, the
    middle time of each condition and concentration, 20 tubes per arm."""
    blocks = [design.assign(background=A3_BASE[0], operator=A3_BASE[1],
                            tf_level=A3_BASE[2])]
    rep1 = design[design["replicate"] == 1]
    middle = (rep1.sort_values("t_sel")
              .groupby(["library", "condition_sel", "titrant_conc"])
              .nth(1))
    for block in A3_ARMS:
        arm = middle.assign(background=block[0], operator=block[1],
                            tf_level=block[2])
        arm["sample"] = arm["sample"] + "_" + "_".join(block)
        blocks.append(arm)
    out = pd.concat(blocks, ignore_index=True)
    out["library"] = [
        _library_name(m, (b, o, t)) for m, b, o, t in
        zip(out["library"], out["background"], out["operator"],
            out["tf_level"])]
    return out


def a3_config(cf, design):
    """A2's config with the A3 design, contexts, background and binding."""
    cf = copy.deepcopy(cf)
    cf["design"] = "../../inputs/design_A3.csv"
    cf["contexts"] = "../../inputs/contexts_A3.yaml"
    cf["backgrounds"] = copy.deepcopy(A3_BACKGROUND)
    cf["binding_data"].update(A3_BINDING)
    libraries = sorted(set(design["library"]))
    marker = {lib: lib.split("_")[0] for lib in libraries}
    for key in ("cfu0", "unassigned_read_fraction"):
        cf[key] = {lib: cf[key][marker[lib]] for lib in libraries}
    # the same reads per tube as A2
    cf["total_num_reads"] = int(round(cf["total_num_reads"] * len(design)
                                      / A2_TUBES))
    return cf


A2_TUBES = 118


def main():
    with open(os.path.join(HERE, "inputs", "base_smoke.yaml")) as f:
        base = yaml.safe_load(f)
    for name, spec in RUNS.items():
        cf = copy.deepcopy(base)
        cf.update(COMMON)
        cf.pop("theta_priors", None)
        cf["theta_component"] = spec["theta_component"]
        priors = {f"theta_{s}_wt_scale": WT_PIN for s in spec["wt_sites"]}
        priors["theta_epi_tau_scale"] = 1.0e-9
        cf["theta_priors"] = priors
        cf["od600"]["calibration"] = "../../inputs/od600_calibration.yaml"
        out = os.path.join(HERE, "runs", name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "tfs_sim_config.yaml"), "w") as f:
            f.write(f"# Stage A, run {name}: written by make_configs.py\n")
            yaml.safe_dump(cf, f, sort_keys=False)
        print(f"wrote runs/{name}/tfs_sim_config.yaml")

        if name == "A2":
            design = pd.read_csv(os.path.join(HERE, "inputs", "design.csv"))
            assert len(design) == A2_TUBES
            a3 = a3_design(design)
            a3.to_csv(os.path.join(HERE, "inputs", "design_A3.csv"),
                      index=False)
            with open(os.path.join(HERE, "inputs", "contexts_A3.yaml"),
                      "w") as f:
                f.write("# A3 contexts: written by make_configs.py\n")
                yaml.safe_dump(A3_CONTEXTS, f, sort_keys=False)
            out = os.path.join(HERE, "runs", "A3")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "tfs_sim_config.yaml"), "w") as f:
                f.write("# Run A3 (plan step 3g): written by "
                        "make_configs.py\n")
                yaml.safe_dump(a3_config(cf, a3), f, sort_keys=False)
            print(f"wrote runs/A3/tfs_sim_config.yaml ({len(a3)} tubes, "
                  f"{a3['library'].nunique()} libraries)")


if __name__ == "__main__":
    main()
