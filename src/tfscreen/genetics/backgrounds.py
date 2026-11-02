"""
Genetic backgrounds: libraries built on a parent that differs from the
reference ``wt_seq`` (planning/dimer-thermo-model.md, step 3d).

A screen can carry the same library design in several backgrounds, each a
parent sequence with a few mutations, cloned and transformed as a library
of its own. Genotypes are named in absolute terms, relative to the
reference ``wt_seq``: A100V made in the K84L/M42I background is
``K84L/M42I/A100V``. The θ model then needs nothing new: a background's
effect is the sum of its mutations' effects, and its own parent is a
measured genotype.

The experiment's one library YAML declares them in an optional
``backgrounds`` block:

.. code-block:: yaml

    backgrounds:
      bgK84L:
        wt_seq: gccagc...ctgtct...       # this background's amplicon DNA
      bgM42I_far:
        outside_mutations: [M42I]       # mutations outside the amplicon
        library_mixture: {single-1: 100, single-2: 100, double-1-2: 1000}

``wt_seq`` is the amplicon in that background (default: the reference's),
so the expected reads match the real plasmid's codons and every
in-amplicon background mutation is named from the sequence.
``outside_mutations`` lists the background's mutations that fall outside
the amplicon, which the reads never show (protein numbering, as in the
genotype names). ``library_mixture`` and ``spiked_seqs`` default to the
reference's, without the spiked origin when the background gives no spikes
of its own. A tube's background is its tube-table ``background`` column
(absent: :data:`REFERENCE`); ``tfs-process-fastq --background`` calls its
reads.
"""

import copy
import re

from tfscreen.util.io.read_yaml import read_yaml

REFERENCE = "reference"
BACKGROUND_COLUMN = "background"

_BACKGROUND_KEYS = {"wt_seq", "outside_mutations", "library_mixture",
                    "spiked_seqs"}
_MUTATION = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")
_SUBSAMPLE_KEYS = ("library_genotypes", "library_subsample",
                   "library_subsample_seed", "library_subsample_depth")

# LibraryManager keys this module sets for a background's config.
NAME_REFERENCE_KEY = "name_reference_seq"
NAME_EXTRA_KEY = "name_extra_mutations"


def _mutation_list(value, where):
    """Validated protein mutations (``K84L``) from a list or 'A/B' text."""
    if value is None:
        return []
    items = value.split("/") if isinstance(value, str) else list(value)
    out = []
    for m in items:
        m = str(m).strip()
        if not _MUTATION.match(m):
            raise ValueError(f"{where}: {m!r} is not a mutation like K84L.")
        if m[0] == m[-1]:
            raise ValueError(f"{where}: {m!r} is not a mutation.")
        out.append(m)
    sites = [_MUTATION.match(m).group(2) for m in out]
    if len(set(sites)) != len(sites):
        raise ValueError(f"{where}: two mutations at one site in {out}.")
    return out


def has_backgrounds(library_config):
    """True when the library YAML declares a ``backgrounds`` block."""
    cf = read_yaml(library_config)
    return bool((cf or {}).get("backgrounds"))


def background_names(library_config):
    """:data:`REFERENCE` and the declared backgrounds, in file order."""
    cf = read_yaml(library_config) or {}
    return [REFERENCE] + [str(k) for k in (cf.get("backgrounds") or {})]


def read_backgrounds(library_config):
    """
    The declared backgrounds, validated.

    Returns
    -------
    dict
        ``{name: {"wt_seq": str or None, "outside_mutations": list,
        "library_mixture": dict or None, "spiked_seqs": list or None}}``.

    Raises
    ------
    ValueError
        On an unknown key, a name that is :data:`REFERENCE` or empty, a
        ``wt_seq`` of the wrong length, a malformed mutation, or a
        background that is the reference itself.
    """
    cf = read_yaml(library_config) or {}
    raw = cf.get("backgrounds") or {}
    if not isinstance(raw, dict):
        raise ValueError("'backgrounds' must map each background name to its "
                         "settings.")
    present = [k for k in _SUBSAMPLE_KEYS if k in cf]
    if raw and present:
        raise ValueError(f"backgrounds cannot be combined with the library "
                         f"subsample keys {present} yet.")
    ref_seq = "".join(str(cf.get("wt_seq", "")).split())
    out = {}
    for name, spec in raw.items():
        name = str(name)
        where = f"background '{name}'"
        if name in ("", REFERENCE) or "|" in name:
            raise ValueError(f"{where}: invalid name (reserved or empty).")
        spec = {} if spec is None else spec
        if not isinstance(spec, dict):
            raise ValueError(f"{where}: settings must be a mapping.")
        extra = sorted(set(spec) - _BACKGROUND_KEYS)
        if extra:
            raise ValueError(f"{where}: unknown keys {extra}; expected "
                             f"{sorted(_BACKGROUND_KEYS)}.")
        wt_seq = spec.get("wt_seq")
        if wt_seq is not None:
            wt_seq = "".join(str(wt_seq).split()).lower()
            if len(wt_seq) != len(ref_seq):
                raise ValueError(f"{where}: wt_seq has {len(wt_seq)} bases; "
                                 f"the reference wt_seq has {len(ref_seq)}.")
        outside = _mutation_list(spec.get("outside_mutations"), where)
        if (wt_seq is None or wt_seq == ref_seq.lower()) and not outside:
            raise ValueError(f"{where} is the reference: give its wt_seq or "
                             f"outside_mutations.")
        out[name] = {"wt_seq": wt_seq, "outside_mutations": outside,
                     "library_mixture": spec.get("library_mixture"),
                     "spiked_seqs": spec.get("spiked_seqs")}
    return out


def background_config(library_config, background=REFERENCE):
    """
    The library config for one background, for ``LibraryManager``.

    The reference returns the YAML as is (without the ``backgrounds``
    block). A declared background swaps in its ``wt_seq``, mixture and
    spikes and tells ``LibraryManager`` to name genotypes against the
    reference sequence plus the background's outside mutations.

    Parameters
    ----------
    library_config : str or dict
    background : str

    Returns
    -------
    dict
    """
    cf = copy.deepcopy(read_yaml(library_config) or {})
    declared = read_backgrounds(cf)
    cf.pop("backgrounds", None)
    if background == REFERENCE:
        return cf
    if background not in declared:
        raise ValueError(f"Background {background!r} is not declared in the "
                         f"library config (declared: "
                         f"{[REFERENCE] + list(declared)}).")
    spec = declared[background]
    ref_seq = "".join(str(cf["wt_seq"]).split())
    if spec["wt_seq"] is not None:
        cf["wt_seq"] = spec["wt_seq"]
    cf[NAME_REFERENCE_KEY] = ref_seq
    cf[NAME_EXTRA_KEY] = list(spec["outside_mutations"])
    if spec["spiked_seqs"] is not None:
        cf["spiked_seqs"] = spec["spiked_seqs"]
    else:
        cf.pop("spiked_seqs", None)
    if spec["library_mixture"] is not None:
        cf["library_mixture"] = dict(spec["library_mixture"])
    elif "library_mixture" in cf and "spiked_seqs" not in cf:
        cf["library_mixture"] = {k: v for k, v in cf["library_mixture"].items()
                                 if k != "spiked"}
    return cf


def background_parent(library_config, background):
    """The absolute genotype name of a background's own parent."""
    from tfscreen.genetics.library_manager import LibraryManager
    from tfscreen.genetics import standardize_genotypes
    if background == REFERENCE:
        return "wt"
    lm = LibraryManager(background_config(library_config, background))
    return standardize_genotypes(lm._convert_to_aa([lm.wt_seq]))[0]
