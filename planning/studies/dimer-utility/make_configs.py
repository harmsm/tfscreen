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

Both draw the truth from the AF3 state table (kcal/mol at 310.15 K) plus
ddG_truth_sd per mutation and state, with wt held at the model defaults
and no pair epistasis in the truth.

Run from this directory:  ~/miniconda3/bin/python make_configs.py
"""

import copy
import os

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


if __name__ == "__main__":
    main()
