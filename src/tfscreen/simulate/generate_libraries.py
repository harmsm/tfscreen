import pandas as pd
from collections import Counter, defaultdict
from itertools import product, combinations
import re

# --- Data Constants (Module Dependencies) ---

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

# --- Functional Modules ---

def _get_aa_counts_from_codon(degen_codon: str) -> Counter:
    """
    Translates a degenerate codon into a Counter of amino acids.
    """
    codon_bases = [DEGEN_BASE_SPECIFIER[b.lower()] for b in degen_codon]
    possible_codons = [''.join(c) for c in product(*codon_bases)]
    amino_acids = [CODON_TO_AA[c] for c in possible_codons]

    return Counter(amino_acids)

def _parse_site_markers(aa_sequence: str,
                        mutated_sites: str,
                        seq_starts_at: int) -> dict:
    """
    Parses input strings to identify mutation sites for each sub-library. This
    will generate a dictionary of library identifiers (`1`, `2`, etc.) to the
    residue number and site. 
    """

    aa_sequence = re.sub(r'\s', '', aa_sequence)
    mutated_sites = re.sub(r'\s', '', mutated_sites)

    sites = defaultdict(list)
    for i, char in enumerate(mutated_sites):
        if not char.isalpha() or not char.isupper():
            sites[char].append({
                'res_num': i + seq_starts_at,
                'wt_aa': aa_sequence[i],
                'index': i
            })

    return sites

def _generate_single_outcomes_library(sites_for_one_lib: list, aa_counts: Counter) -> pd.DataFrame:
    """
    Generates all possible single outcomes (mutant and wt) for one sub-library.
    """
    records = []
    for site in sites_for_one_lib:
        for mut_aa, count in aa_counts.items():
            is_wt_outcome = (mut_aa == site['wt_aa'])
            genotype = f"{site['wt_aa']}{site['res_num']}{mut_aa}"
            records.append({
                'genotype': genotype,
                'degeneracy': count,
                'res_num': site['res_num'],
                'is_wt': is_wt_outcome
            })
    return pd.DataFrame(records)

def _combine_outcomes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to perform the cross-join and combination logic.
    """

    merged = pd.merge(df1, df2, how='cross', suffixes=('_1', '_2'))
    merged = merged[merged['res_num_1'] < merged['res_num_2']].copy()
    
    merged['degeneracy'] = merged['degeneracy_1'] * merged['degeneracy_2']
    merged['genotype'] = merged['genotype_1'] + '/' + merged['genotype_2']
    merged['num_muts'] = (~merged['is_wt_1']).astype(int) + (~merged['is_wt_2']).astype(int)

    # Return the intermediate columns needed for later steps.
    return merged[[
        'genotype', 'degeneracy', 'num_muts', 
        'genotype_1', 'genotype_2', 'is_wt_1', 'is_wt_2'
    ]]


def generate_libraries(aa_sequence: str,
                       mutated_sites: str,
                       seq_starts_at: int=1,
                       degen_codon: str="nnt",
                       internal_doubles=False) -> pd.DataFrame:
    """
    Generate a DataFrame of mutant libraries from a configuration dictionary.

    This function serves as the main orchestrator for simulating the creation
    of mutant libraries. It mimics an experimental pipeline where specific sites
    in a protein are targeted for mutagenesis using a degenerate codon. The
    function can generate libraries of single mutants, cross-library double
    mutants (combinations of mutations from different sub-libraries), or
    internal double mutants (combinations of mutations within the same
    sub-library). The output DataFrame enumerates every expected genotype, its
    library of origin, and its expected frequency based on codon degeneracy.

    Parameters
    ----------
    config : dict
        A dictionary specifying all parameters for the library generation. It
        should contain the following keys:
        - 'aa_sequence' (str): The wild-type amino acid sequence.
        - 'mutated_sites' (str): The sequence modified with non-uppercase
          characters to mark sites for mutation. Each unique character
          defines a sub-library (e.g., '1', '2').
        - 'seq_starts_at' (int): The residue number corresponding to the first
          amino acid in `aa_sequence`.
        - 'degen_codon' (str): The degenerate codon used at each marked site,
          e.g., 'nnt' or 'nnk'.
        - 'internal_doubles' (bool): Controls double mutant generation logic.
          If False, creates cross-library doubles (e.g., a mutation from
          library '1' paired with one from '2'). If True, creates all
          internal doubles (e.g., two distinct mutations from within
          library '1').

    Returns
    -------
    pandas.DataFrame
        A DataFrame listing all generated genotypes, sorted by library and
        degeneracy. It has the following columns:
        - 'library_origin' (str): The library from which the genotype
          originates (e.g., 'single-1', 'double-1-2').
        - 'genotype' (str): The genotype string. 'wt' for wild-type, 'A10G'
          for single mutants, and 'S30A/V75G' for double mutants.
        - 'degeneracy' (int): The expected frequency of the genotype, accounting
          for codon degeneracy.

    Examples
    --------
    >>> config = {
    ...     'aa_sequence': "ASKV",
    ...     'mutated_sites': "A1K2",
    ...     'seq_starts_at': 10,
    ...     'internal_doubles': False,
    ...     'degen_codon': 'GCN' # Encodes Alanine 4 times
    ... }
    >>> library_df = generate_libraries(config)
    >>> print(library_df)
      library_origin   genotype  degeneracy
    0     double-1-2  S11A/V13A          16
    1       single-1       S11A           4
    2       single-2       V13A           4

    """
    
    site_markers = _parse_site_markers(aa_sequence,
                                       mutated_sites,
                                       seq_starts_at)
    
    aa_counts = _get_aa_counts_from_codon(degen_codon)
    
    all_results = []
    single_outcomes = {
        lib_id: _generate_single_outcomes_library(sites, aa_counts)
        for lib_id, sites in site_markers.items()
    }

    # 3. Process combined libraries
    combination_dfs = []
    if internal_doubles:
        for lib_id, df in single_outcomes.items():
            if len(df['res_num'].unique()) > 1:
                combined = _combine_outcomes(df, df)
                combined['library_origin'] = f'internal-double-{lib_id}'
                combination_dfs.append(combined)
    else:
        lib_ids = list(single_outcomes.keys())
        for id1, id2 in combinations(lib_ids, 2):
            combined = _combine_outcomes(single_outcomes[id1], single_outcomes[id2])
            combined['library_origin'] = f'double-{id1}-{id2}'
            combination_dfs.append(combined)
            
    if combination_dfs:
        full_combined_df = pd.concat(combination_dfs, ignore_index=True)
        for num_muts_val in [0, 1, 2]:
            subset_df = full_combined_df[full_combined_df['num_muts'] == num_muts_val].copy()
            if subset_df.empty:
                continue
            if num_muts_val == 0:
                subset_df = subset_df.groupby('library_origin', as_index=False)['degeneracy'].sum()
                subset_df['genotype'] = 'wt'
            elif num_muts_val == 1:
                subset_df['genotype'] = subset_df.apply(
                    lambda row: row['genotype_1'] if not row['is_wt_1'] else row['genotype_2'],
                    axis=1)
            all_results.append(subset_df[['library_origin', 'genotype', 'degeneracy']])

    # 4. Process the pure single-mutant libraries AND THEIR WILDTYPES
    for lib_id, df in single_outcomes.items():
        # Add the mutants from this library
        mutants_df = df[~df['is_wt']].copy()
        mutants_df['library_origin'] = f'single-{lib_id}'
        all_results.append(mutants_df[['library_origin', 'genotype', 'degeneracy']])

        # --- CORRECTED LOGIC FOR SINGLE-LIBRARY WT ---
        # Calculate and add the 'wt' for this library
        wt_outcomes_df = df[df['is_wt']].copy()
        if not wt_outcomes_df.empty:
            # The total wt count is the SUM of all individual wt outcome counts
            wt_count = wt_outcomes_df['degeneracy'].sum() # <-- THE FIX IS HERE (.prod() -> .sum())
            wt_single_df = pd.DataFrame([{
                'library_origin': f'single-{lib_id}',
                'genotype': 'wt',
                'degeneracy': wt_count
            }])
            all_results.append(wt_single_df)

    # 5. Concatenate everything for the final result
    if not all_results:
        return pd.DataFrame(columns=['library_origin', 'genotype', 'degeneracy'])
        
    final_df = pd.concat(all_results, ignore_index=True)
    return final_df.sort_values(
        by=['library_origin', 'degeneracy'], ascending=[True, False]
    ).reset_index(drop=True)