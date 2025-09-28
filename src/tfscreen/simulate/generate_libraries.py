
from tfscreen.util import (
    standardize_genotypes,
    set_categorical_genotype
)

import pandas as pd
import numpy as np

from itertools import (
    product,
)
import re
from typing import (
    List,
)

DEGEN_BASE_SPECIFIER = {
    "a": "a", "c": "c", "g": "g", "t": "t",
    "r": "ag", "y": "ct", "m": "ac", "k": "gt", "s": "cg", "w": "at",
    "h": "act", "b": "cgt", "v": "acg", "d": "agt", "n": "acgt"
}

CODON_TO_AA = {
    'ttt': 'F', 'tct': 'S', 'tat': 'Y', 'tgt': 'C',
    'ttc': 'F', 'tcc': 'S', 'tac': 'Y', 'tgc': 'C',
    'tta': 'L', 'tca': 'S', 'taa': '*', 'tga': '*',
    'ttg': 'L', 'tcg': 'S', 'tag': '*', 'tgg': 'W',
    'ctt': 'L', 'cct': 'P', 'cat': 'H', 'cgt': 'R',
    'ctc': 'L', 'ccc': 'P', 'cac': 'H', 'cgc': 'R',
    'cta': 'L', 'cca': 'P', 'caa': 'Q', 'cga': 'R',
    'ctg': 'L', 'ccg': 'P', 'cag': 'Q', 'cgg': 'R',
    'att': 'I', 'act': 'T', 'aat': 'N', 'agt': 'S',
    'atc': 'I', 'acc': 'T', 'aac': 'N', 'agc': 'S',
    'ata': 'I', 'aca': 'T', 'aaa': 'K', 'aga': 'R',
    'atg': 'M', 'acg': 'T', 'aag': 'K', 'agg': 'R',
    'gtt': 'V', 'gct': 'A', 'gat': 'D', 'ggt': 'G',
    'gtc': 'V', 'gcc': 'A', 'gac': 'D', 'ggc': 'G',
    'gta': 'V', 'gca': 'A', 'gaa': 'E', 'gga': 'G',
    'gtg': 'V', 'gcg': 'A', 'gag': 'E', 'ggg': 'G'
}

def _validate_inputs(aa_sequence,
                     mutated_sites,
                     seq_starts_at,
                     lib_keys):
    
    # Validate and clean up aa_sequence and mutated_sites, which should be 
    # strings of the same length
    ws = re.compile(r"\s")
    clean_aa = ws.sub("",aa_sequence).upper()
    clean_mut = ws.sub("",mutated_sites).upper()

    # Make sure lengths are the same
    if len(clean_aa) != len(clean_mut):
        err = "aa_sequence and mutated_sites must be strings of the same length\n"
        raise ValueError(err)
        
    # Validate seq_starts_at 
    try:
        seq_starts_at = int(seq_starts_at)
    except (ValueError,TypeError):
        err = "seq_starts_at must be an integer\n"
        raise ValueError(err)

    # Clean up whitespace in lib_keys
    lib_keys = [k.strip() for k in lib_keys]

    return clean_aa, clean_mut, seq_starts_at, lib_keys

def _find_mut_sites(clean_aa,clean_mut,seq_starts_at):

    # Extract wildtype amino acids and sites keyed to blocks
    wt_aa = {}
    sites = {}

    resid_number = seq_starts_at
    for aa, block in zip(list(clean_aa),list(clean_mut)):

        if aa == block: 
            resid_number += 1
            continue
        
        if block not in wt_aa:
            wt_aa[block] = []
            sites[block] = []

        wt_aa[block].append(aa)
        sites[block].append(resid_number)
        resid_number += 1

    return wt_aa, sites

def _expand_degen_codon(degen_codon):

    degen_bases = [DEGEN_BASE_SPECIFIER[b.lower()] for b in degen_codon]
    possible_muts = [CODON_TO_AA[''.join(c)] for c in product(*degen_bases)]
    
    return possible_muts

def _generate_singles(wt_aa,sites,possible_muts):

    # Go over all single blocks
    lib_genotypes = {}
    for block in wt_aa.keys():

        # This appends all possible mutations to all wildtype aa/site strings. 
        # We'll worry about things like H29H later. 
        wt_base = [f"{wt_aa[block][i]}{sites[block][i]:d}"
                   for i in range(len(wt_aa[block]))]
        muts = ["".join(p) for p in product(wt_base,possible_muts)]
        lib_genotypes[f"single-{block}"] = muts

    return lib_genotypes

def _generate_inter_block(block_a,block_b,lib_genotypes):

    # Grab the singles libraries from each block because we're cross
    # combining two blocks. 
    geno_a = lib_genotypes[f"single-{block_a}"]
    geno_b = lib_genotypes[f"single-{block_b}"]

    # make all possible double mutants
    doubles = ["/".join(m) for m in product(geno_a,geno_b)]
    
    return doubles

def _generate_intra_block(wt_aa,sites,block,possible_muts):
    
    # Pre-calculate all possible muts at all sites
    muts_at_site = []
    for i in range(len(sites[block])):
        wt_base = f"{wt_aa[block][i]}{sites[block][i]}"
        muts = ["".join(p) for p in product([wt_base],possible_muts)]
        muts_at_site.append(muts)
    
    # Go through all pairs of **sites**. This prevents things like
    # H29A/H29C. We'll combine mutations at (29,30),(29,31) ... 
    doubles = []
    for i in range(len(sites[block])):                
        for j in range(i+1,len(sites[block])):

            # Make all possible combinations of mutations at sites i and j
            mut_combos = ["/".join(m) for m in product(muts_at_site[i],
                                                       muts_at_site[j])]
            doubles.extend(mut_combos)

    # Record double mutants
    return doubles 

def _build_final_df(lib_genotypes):
    
    dfs = []

    # Go through each key
    for key in lib_genotypes:

        # Clean up the genotypes. This does things like convert H29H to wt.
        clean_genotypes = standardize_genotypes(lib_genotypes[key])

        # Grab unique genotypes and their counts from the library. We want 
        # these to account for degeneracy when using the libraries for 
        # transformation etc. 
        genotypes, counts = np.unique(clean_genotypes,return_counts=True)

        # Build a pandas DataFrame for the library
        out_dict = {}
        out_dict["library_origin"] = np.full(len(genotypes),key)
        out_dict["genotype"] = genotypes
        out_dict["degeneracy"] = counts
        out_dict["weight"] = out_dict["degeneracy"]/np.sum(out_dict["degeneracy"])

        dfs.append(pd.DataFrame(out_dict))

    # Build a single large genotype dataframe. Set the genotypes to categorical
    # (allowing useful sorting). 
    df = pd.concat(dfs,ignore_index=True)
    df = set_categorical_genotype(df)
    df = df.sort_values(["genotype","library_origin"]).reset_index(drop=True)

    return df

def generate_libraries(
    aa_sequence: str,
    mutated_sites: str,
    degen_codon: str,
    seq_starts_at: int,
    lib_keys: List[str]
) -> pd.DataFrame:
    """
    Generate a DataFrame of all possible mutant library genotypes.

    This function simulates the experimental creation of genetic libraries by
    enumerating all possible single and double mutants derived from specific
    "blocks" of residues targeted for mutagenesis with a degenerate codon.

    Parameters
    ----------
    aa_sequence : str
        The wildtype amino acid sequence, allowing for newlines and spaces.
    mutated_sites : str
        A string of the same length as `aa_sequence` where mutation sites
        are marked with a non-uppercase character. Each unique character
        (e.g., '1', '2', 'a') different from the sequence in aa_sequence
        defines a distinct "block" for mutagenesis.
    degen_codon : str
        The degenerate codon used for mutagenesis (e.g., "nnt", "nnk").
    seq_starts_at : int
        The residue number of the first amino acid in `aa_sequence`.
    lib_keys : list[str]
        A list of the specific libraries to generate and include in the final
        output. Examples:
        - 'single-1': All single mutants from block '1'.
        - 'double-1-2': All inter-block doubles between blocks '1' and '2'.
        - 'double-1-1': All intra-block doubles within block '1'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all specified genotypes, with the columns:
        - 'library_origin' (str): The key from `lib_keys` for the genotype.
        - 'genotype' (str): The standardized genotype string (e.g., 'wt', 
          'A10G', 'V30L/I45M').
        - 'degeneracy' (int): The number of times that specific genotype is
          expected to be generated based on the degenerate codon and 
          combinations of mutations made. 

    Examples
    --------
    >>> config = {
    ...     "aa_sequence": "ASKV",
    ...     "mutated_sites": "A1K2",
    ...     "degen_codon": "tgg",  # Only encodes Tryptophan (W)
    ...     "seq_starts_at": 10,
    ...     "lib_keys": ["single-1", "single-2", "double-1-2"]
    ... }
    >>> df = generate_libraries(**config)
    >>> print(df)
       library_origin      genotype  degeneracy
    0    double-1-2          wt           1
    1    double-1-2        S11W           1
    2    double-1-2        V13W           1
    3    double-1-2   S11W/V13W           1
    4      single-1          wt           1
    5      single-1        S11W           1
    6      single-2          wt           1
    7      single-2        V13W           1

    """
    
    # Validate inputs
    clean_aa, clean_mut, seq_starts_at, lib_keys = _validate_inputs(
        aa_sequence,
        mutated_sites,
        seq_starts_at,
        lib_keys
    )
    
    # Parse amino acid sequence and mutated sites to identify blocks of 
    # mutations and figure out all possible muts encoded by degenerate codon
    wt_aa, sites = _find_mut_sites(clean_aa,clean_mut,seq_starts_at)

    # Expand the degnerate codon to all possible amino acids it might encode.     
    possible_muts = _expand_degen_codon(degen_codon)

    # Pre-generate single mutant libraries. This also initializes the 
    # lib_genotypes dictionary that keys lib_key to genotypes. 
    lib_genotypes = _generate_singles(wt_aa,
                                      sites,
                                      possible_muts)
    
    # Go through requested library keys 
    for lib_key in lib_keys:
    
        # We already generated singles above. Don't repeat if we already made 
        # this lib_key
        if lib_key in lib_genotypes: 
            continue
    
        # If this is true, there is a lib_key that starts with single but does
        # not follow the pattern single-x, with x being part of the
        # mutated_sites string
        if lib_key.startswith("single"):
            err = f"{lib_key} should be formatted like 'single-x' where 'x' is \n"
            err += "the library character in mutated_sites\n"
            raise ValueError(err)

        # Split the lib_key
        columns = lib_key.split("-")

        # Make sure it looks like double-x-y
        if len(columns) != 3 or columns[0] != "double":
            err = f"could not parse '{lib_key}'. Expecting something like\n"
            err += "double-x-y, where 'x' and 'y' are in the mutated_sites\n"
            err += "string.\n"
            raise ValueError(err)

        # make sure we can find first library specified
        if columns[1] not in wt_aa:
            err = f"could not match mutant identifier '{columns[1]}' from\n"
            err += f"'{lib_key}' to mutated_sites\n"
            raise ValueError(err)

        # make sure we can find the second library specified
        if columns[2] not in wt_aa:
            err = f"could not match mutant identifier '{columns[2]}' from\n"
            err += f"'{lib_key}' to mutated_sites\n"
            raise ValueError(err)

        # Grab the relevant blocks
        block_a = columns[1]
        block_b = columns[2]

        # between block mutant pairs
        if block_a != block_b:
            lib_genotypes[lib_key] = _generate_inter_block(block_a,
                                                           block_b,
                                                           lib_genotypes)

        # within-block mutant pairs
        else:
            lib_genotypes[lib_key] = _generate_intra_block(wt_aa,
                                                           sites,
                                                           block_a,
                                                           possible_muts)

    # Remove library keys that the user didn't request. (We generated all
    # singles, for example)
    final_lib_genotypes = {}
    for k in lib_genotypes:
        if k in lib_keys:
            final_lib_genotypes[k] = lib_genotypes[k]

    # build clean final dataframe
    df = _build_final_df(final_lib_genotypes)
        
    return df