"""
Simulated binding data for the thermodynamic theta components.

Spiked stratified/random binding used to build a wt-only stratified pool,
which the thermodynamic models could not draw (a numpyro plate of size 0),
and in-library binding read Hill columns these models do not have.  Spiked
genotypes now keep their own model-generated theta, and every binding theta
is evaluated from the drawn parameters at the binding concentrations.  These
tests simulate a tiny library with each binding mode and check that the
binding table exists, is finite, equals the theta computed from the written
K truth at the binding concentrations, and agrees with the growth theta at
shared concentrations.
"""
import inspect
import warnings

import jax
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers

from tfscreen.simulate.binding_data import generate_binding_df
from tfscreen.simulate.library_binding_data import generate_library_binding_df
from tfscreen.simulate.library_prediction import library_prediction
from tfscreen.simulate.sample_theta import sample_theta_prior
from tfscreen.simulate.sim_data_class import build_sim_data
from tfscreen.tfmodel.generative.registry import model_registry

# Two nnt-degenerate codons (residues 29 and 62 of lac repressor) and two
# spiked single mutants: a few hundred genotypes, drawn in about a second.
_WT = ("gccagccacgtttctgcgaaaacgcgggaaaaagtggaagcggcgatggcggagctgaattacattcccaa"
       "ccgcgtggcacaacaactggcgggcaaacagtcgttgctgattggcgttgccacctccagtctggccctg"
       "cacgcgccgtcgcaaattgtcgcggcgattaaatctcgcgccgatcaactgggtgccagcgtggtggtgt"
       "cgatg")


def _site_string(char_at):
    s = ["."] * len(_WT)
    for pos, c in char_at.items():
        for k in range(3):
            s[pos + k] = c
    return "".join(s)


def _spike(codon_pos, codon):
    return _WT[:codon_pos] + codon + _WT[codon_pos + 3:]


_DEGEN = _site_string({6: "n", 105: "n"}).replace("nnn", "nnt")
_TILES = _site_string({6: "1", 105: "2"})

# Binding concentrations: 0 and 0.01 are growth concentrations, the rest
# are not, so the theta must be evaluated rather than looked up.
BINDING_CONCS = [0.0, 0.0003, 0.01, 0.05, 2.0]
GROWTH_CONCS = [0.0, 0.001, 0.01, 1.0]

_C4_STRUCTS = ["H", "HD", "L", "LE2"]


def _config(component, choose_by, thermo_data=None):
    cf = {
        "reading_frame": 0,
        "first_amplicon_residue": 27,
        "wt_seq": _WT,
        "degen_sites": _DEGEN,
        "tiles": _TILES,
        "tile_combos": ["single-1", "single-2", "double-1-2"],
        "spiked_seqs": [_spike(6, "gct"), _spike(105, "gct"),
                        _spike(6, "tgg")],
        "theta_component": component,
        "condition_blocks": [{
            "library": "kanR", "titrant_name": "iptg",
            "titrant_conc": GROWTH_CONCS,
            "condition_pre": "pre", "t_pre": 30,
            "condition_sel": "sel", "t_sel": [100],
        }],
        "growth": {"pre": {"m": 0.0, "b": 0.01},
                   "sel": {"m": -0.01, "b": 0.0}},
        "dk_geno_zero": True,
        "seed": 3,
        "binding_data": {
            "titrant_name": "iptg",
            "titrant_conc": BINDING_CONCS,
            "noise": 0.0,
            "spiked_binding": {"choose_by": choose_by, "num": 2},
            "library_binding": {"choose_by": choose_by, "num": 3},
        },
    }
    if thermo_data is not None:
        cf["thermo_data"] = thermo_data
    return cf


def _ddG_csv(path):
    rng = np.random.default_rng(3)
    muts = ["H29A", "H29W", "L62A"]
    df = pd.DataFrame(rng.normal(0, 1, (len(muts), len(_C4_STRUCTS))),
                      columns=_C4_STRUCTS)
    df.insert(0, "mut", muts)
    df.to_csv(path, index=False)
    return str(path)


def _theta_from_truth(component, params, genotypes, concs):
    """Theta at ``concs`` from the K columns of ``parameters_df`` (G, C)."""
    module = model_registry["theta"][component]
    hyper = module.get_hyperparameters()
    names = [n for n in inspect.signature(module._compute_theta).parameters
             if n.startswith("ln_K")]
    p = params.assign(genotype=params["genotype"].astype(str)).set_index("genotype")
    ks = []
    for n in names:
        v = p.loc[genotypes, n].to_numpy(dtype=float)
        ks.append(v[None, :] if n.lower().endswith("e") else v)
    # Both families take M (mM times theta_conc_unit_scale).
    scale = hyper.get("theta_conc_unit_scale", 1.0)
    theta = module._compute_theta(*ks, np.asarray(concs, dtype=float) * scale,
                                  hyper["theta_tf_total_M"])
    return np.asarray(theta)[0].T


def _simulate(component, choose_by, tmp_path):
    td = (_ddG_csv(tmp_path / "ddG.csv")
          if component.endswith("PddG") else None)
    cf = _config(component, choose_by, thermo_data=td)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = library_prediction(cf, return_mutation_truth=True,
                                 return_binding_theta=True)
    return cf, out


CASES = [
    "thermo.O2_C4_K3_U0_a.PK",
    "thermo.O2_C12_K5_U0_a.PK",
    "thermo.O2_C4_K3_U0_a.PddG",
]


@pytest.mark.parametrize("choose_by", ["stratified", "random"])
@pytest.mark.parametrize("component", CASES)
def test_spiked_binding_keeps_own_theta(component, choose_by, tmp_path):
    cf, out = _simulate(component, choose_by, tmp_path)
    library_df, _, genotype_theta_df, params, binding_theta_df, _, lbt = out

    spiked = set(library_df.loc[library_df["library_origin"] == "spiked",
                                "genotype"])
    assert binding_theta_df is not None
    genos = list(pd.unique(binding_theta_df["genotype"]))
    assert len(genos) == 2 and set(genos) <= spiked
    assert np.isfinite(binding_theta_df["theta_true"]).all()

    # Binding theta is the theta of the genotype's own K truth.
    wide = binding_theta_df.pivot(index="genotype", columns="titrant_conc",
                                  values="theta_true").loc[genos, BINDING_CONCS]
    expected = _theta_from_truth(component, params, genos, BINDING_CONCS)
    np.testing.assert_allclose(wide.to_numpy(), expected, rtol=1e-4, atol=1e-6)

    # At concentrations shared with growth, binding equals growth theta.
    gt = genotype_theta_df.assign(genotype=genotype_theta_df["genotype"].astype(str))
    for g in genos:
        for c in (0.0, 0.01):
            growth = gt.loc[(gt["genotype"] == g) & (gt["titrant_conc"] == c),
                            "theta"].item()
            assert wide.loc[g, c] == pytest.approx(growth, rel=1e-4, abs=1e-6)

    # The measured table (noise 0) reproduces it.
    measured = generate_binding_df(cf["binding_data"], np.random.default_rng(0),
                                   binding_theta_df)
    assert len(measured) == 2 * len(BINDING_CONCS)
    np.testing.assert_allclose(measured["theta_obs"],
                               binding_theta_df["theta_true"])


@pytest.mark.parametrize("choose_by", ["stratified", "random"])
@pytest.mark.parametrize("component", CASES)
def test_library_binding_uses_simulated_theta(component, choose_by, tmp_path):
    cf, out = _simulate(component, choose_by, tmp_path)
    library_df, _, _, params, _, _, lbt = out
    assert lbt is not None
    assert np.isfinite(lbt["theta"]).all()

    spiked = list(pd.unique(library_df.loc[
        library_df["library_origin"] == "spiked", "genotype"]))
    growth_df = pd.DataFrame({"genotype": library_df["genotype"].unique()})
    bdf, manifest = generate_library_binding_df(
        cf["binding_data"]["library_binding"], "iptg", BINDING_CONCS, 0.0,
        params, growth_df, spiked_genotypes=spiked,
        rng=np.random.default_rng(0), theta_df=lbt)

    genos = list(manifest["genotype"])
    assert len(genos) == 3
    assert not set(genos) & (set(spiked) | {"wt"})
    assert len(bdf) == 3 * len(BINDING_CONCS)
    assert np.isfinite(bdf["theta_obs"]).all()
    wide = bdf.pivot(index="genotype", columns="titrant_conc",
                     values="theta_obs").loc[genos, BINDING_CONCS]
    expected = _theta_from_truth(component, params, genos, BINDING_CONCS)
    np.testing.assert_allclose(wide.to_numpy(), expected, rtol=1e-4, atol=1e-6)


def test_hill_library_binding_theta_matches_hill_columns(tmp_path):
    """For hill_geno the evaluated theta equals the Hill-column fallback."""
    cf = _config("hill_geno", "stratified")
    del cf["binding_data"]["spiked_binding"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        library_df, _, _, params, _, lbt = library_prediction(
            cf, return_binding_theta=True)
    growth_df = pd.DataFrame({"genotype": library_df["genotype"].unique()})
    args = (cf["binding_data"]["library_binding"], "iptg", BINDING_CONCS, 0.0,
            params, growth_df)
    from_theta, _ = generate_library_binding_df(
        *args, spiked_genotypes=None, rng=np.random.default_rng(0),
        theta_df=lbt)
    from_hill, _ = generate_library_binding_df(
        *args, spiked_genotypes=None, rng=np.random.default_rng(0))
    pd.testing.assert_frame_equal(from_theta, from_hill, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Zero-mutation (wt-only) libraries
# ---------------------------------------------------------------------------

ALL_THERMO = sorted(k for k in model_registry["theta"] if k.startswith("thermo."))


def _wt_only_sim_data(module):
    sd = build_sim_data(pd.DataFrame({"genotype": ["wt"]}),
                        pd.DataFrame({"titrant_conc": [0.0, 0.01, 1.0]}),
                        skip_pairs=True)
    names = getattr(module, "STRUCTURE_NAMES", None)
    if names is None:
        return sd
    S = len(names)
    nn = "PnnC" in module.__name__
    return sd.replace(
        num_struct=S, struct_names=tuple(names),
        struct_features=np.zeros((0, S, 60) if nn else (0, S), np.float32),
        struct_n_chains=np.ones(S, np.int32) if nn else None)


def test_every_thermo_component_listed():
    assert len(ALL_THERMO) == 12


@pytest.mark.parametrize("component", ALL_THERMO)
def test_zero_mutation_define_model_and_guide(component):
    module = model_registry["theta"][component]
    sd = _wt_only_sim_data(module)
    assert sd.num_mutation == 0

    theta, _ = sample_theta_prior(component, sd, jax.random.PRNGKey(0),
                                  force_prior_predictive=True)
    assert theta.shape == (1, 3)
    assert np.isfinite(theta).all()

    priors = module.ModelPriors(**module.get_hyperparameters())
    with handlers.seed(rng_seed=0):
        param = module.guide("theta", sd, priors)
    assert np.isfinite(np.asarray(module.run_model(param, sd))).all()


def test_stratified_pool_draws_thermo_component():
    """The wt-only stratified pool (the old failure) now draws."""
    from tfscreen.simulate.sample_theta import sample_theta_stratified
    b, g = sample_theta_stratified(
        "thermo.O2_C12_K5_U0_a.PK",
        binding_sample_df=pd.DataFrame({"titrant_conc": BINDING_CONCS}),
        growth_sample_df=pd.DataFrame({"titrant_conc": GROWTH_CONCS}),
        rng_key=jax.random.PRNGKey(0), n_select=2, pool_size=4)
    assert b.shape == (2, len(BINDING_CONCS)) and np.isfinite(b).all()
    assert g.shape == (2, len(GROWTH_CONCS)) and np.isfinite(g).all()
