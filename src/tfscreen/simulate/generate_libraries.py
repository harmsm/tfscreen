"""
Functions for generating mutant libraries based on degenerate codon information.
"""

from tfscreen import data

import pandas as pd
from tqdm.auto import tqdm

import string
import itertools
import re

def _build_genotype_df(all_genotypes):
    """
    Build a DataFrame with genotype information for all clones.

    Parameters
    ----------
    all_genotypes : list
        List of all genotypes with full genotype information.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with genotype, mutation, and site information.
    """
    
    unique_genotypes = set(all_genotypes)
    
    out_dict = {"genotype":[],
                "num_muts":[],
                "site_1":[],
                "site_2":[],
                "mut_1":[],
                "mut_2":[],
                "wt_1":[],
                "wt_2":[]}
                
    for genotype in unique_genotypes:
        
        mut = genotype.split("/")
        
        if len(mut) == 1:
            if mut[0] == "" or mut[0] == "wt":
                name = "wt"
                num_muts = 0
                m1 = None
                m2 = None
                s1 = None
                s2 = None
                w1 = None
                w2 = None
            else:
                name = genotype
                num_muts = 1
                m1 = mut[0][-1]
                m2 = None
                s1 = int(mut[0][1:-1])
                s2 = None
                w1 = mut[0][0]
                w2 = None
            
        # two mutations
        elif len(mut) == 2:
            name = genotype
            num_muts = 2

            m1 = mut[0][-1]
            m2 = mut[1][-1]
            s1 = int(mut[0][1:-1])
            s2 = int(mut[1][1:-1])
            w1 = mut[0][0]
            w2 = mut[1][0]
    
        else:
            err = "more than two mutations\n"
            raise ValueError(err)
    
        out_dict["genotype"].append(name)
        out_dict["num_muts"].append(num_muts)
        out_dict["site_1"].append(s1)
        out_dict["site_2"].append(s2)
        out_dict["mut_1"].append(m1)
        out_dict["mut_2"].append(m2)
        out_dict["wt_1"].append(w1)
        out_dict["wt_2"].append(w2)
    
    df = pd.DataFrame(out_dict)
    
    df = df.sort_values(by=["num_muts","site_1","site_2","mut_1","mut_2"],axis=0)
    df = df.reset_index(drop=True)
    df.index = df["genotype"]

    return df


def _create_mut_aa_list(degen_codon):
    """
    Get a list of the amino acids encoded by a possibly degenerate codon.
    """

    amino_acids = []
    for b0 in list(data.degen_base_specifier[degen_codon[0]]):
        for b1 in list(data.degen_base_specifier[degen_codon[1]]):
            for b2 in list(data.degen_base_specifier[degen_codon[2]]):
                aa = data.codon_to_aa[f"{b0}{b1}{b2}"]
                amino_acids.append(aa)

    return amino_acids


def _get_libs_to_build(mutated_sites,
                       max_num_combos=2,
                       internal_doubles=False):
    """
    Build a list of all combos of up to max_num_combos sub libraries from a 
    string of mutated sites.

    Parameters
    ----------
    mutated_sites : str
        string holding library information. This should be an amino acid 
        sequence. Any non-uppercase letter is treated as a library with specific
        sites to mutate. The following has two sublibraries, 1 and 2. 
            MAST111QRVT222MNQR...
    max_num_combos : int, default = 2
        maximum number of combinations to make. 1 means do each library only 
        individual; 2 means make pairwise combos; 3 means make three-way, etc. 
    internal_doubles : bool, default=False
        if True, make double mutants within a library block. for the 
        mutated_sites example above, this would make all combos within 1 and
        within 2, in addition to 1, 2, and (1,2). 

    Returns
    -------
    lib_id : list
        list of all sub-libraries seen. would be ["1","2"] for the mutated_sites
        example above
    libraries_to_build : list
        list of tuples of libraries to build. would be [("1",),("2",),("1","2")]
        for the mutated_sites example above if internal_doubles is False. Would
        be [("1",),("1","1"),("2",),("2","2"),("1","2")] if internal_doubles is
        True.
    """

    # remove all white space from mutated_sites            
    mutated_sites = re.sub(r'\s+','',mutated_sites)
    
    # Record all non-uppercase letter values in mutated_sites
    lib_id = []
    for s in mutated_sites:
        if s in string.ascii_uppercase:
            lib_id.append(None)
        else:
            lib_id.append(s)
    
    # Get unique non-uppercase letter values and sort
    all_libs = list(set([s for s in lib_id if s is not None]))
    all_libs.sort()
    
    # Build all combos of the libraries
    libraries_to_build = []
    for k in range(max_num_combos):
    
        # Don't build combos that are higher-order than our number of sub 
        # libraries
        if k + 1 > len(all_libs):
            break
    
        # Build out all combos of this order from the libraries
        to_build_combo = list(itertools.combinations(all_libs,r=(k+1)))
        libraries_to_build.extend(to_build_combo)
    
        # Build internal doubles if requested
        if k == 0 and internal_doubles:
            for i in range(len(to_build_combo)):
                libraries_to_build.append((to_build_combo[i][0],
                                           to_build_combo[i][0]))
    

    return lib_id, libraries_to_build


def generate_libraries(aa_sequence,
                       mutated_sites,
                       seq_starts_at=1,
                       max_num_combos=2,
                       internal_doubles=False,
                       degen_codon="nnt"):
    """
    Take an amino acid sequence and description of sites to mutate and generate
    a list of libraries. Libraries are returned at the amino acid mutation level
    and correspond to all clones encoded by the degenerate codons at the sites 
    indicated. This means the same genotype may occur multiple times.  

    Parameters
    ----------
    aa_sequence : str
        Amino acid sequence as uppercase letters. Whitespace and line-breaks are allowed.
        Example: 'MASTRKEQRVTLFGMNQR...'
    mutated_sites : str
        Amino acid sequence as uppercase letters with mutated sites replaced by library
        identifier characters. Whitespace and line-breaks are allowed.
        Example: 'MAST111QRVT222MNQR...'
        This specifies there are two sublibraries "1" and "2" that replace the amino
        codon for that amino acid with 'degen_codon'. A library id can be any non-uppercase-letter character.
    seq_starts_at : int, default=1
        Number of the amino acid corresponding to the first position in aa_sequence.
    max_num_combos : int, default=2
        Maximum number of combinations to make. 1 means do each library only individually;
        2 means make pairwise combos; 3 means make three-way, etc.
    internal_doubles : bool, default=False
        If True, make double mutants within a library block. For the mutated_sites example above,
        this would make all combos within 1 and within 2, in addition to 1, 2, and (1,2).
    degen_codon : str, default="nnt"
        Use this codon at each site that is mutated.

    Returns
    -------
    lib_clone_dict : dict
        Dictionary keying library combinations to a list of clone names. For the examples above,
        this would have keys like ("1",), ("2",), ("1","2",). Values are lists of genotype strings
        like ['R5A', 'R5C', ..., 'R5A/L12A']. The wildtype will have the key "wt".
    genotype_df : pandas.DataFrame
        DataFrame with genotype, mutation, and site information for all clones.
    """

    # Get amino acid sequence, stripping out white space
    aa_sequence = re.sub(r'\s+','',aa_sequence)

    # Get set of library combinations to build [(1,), (2,), (1,2), etc.]
    lib_id, libraries_to_build = _get_libs_to_build(mutated_sites=mutated_sites,
                                                    max_num_combos=max_num_combos,
                                                    internal_doubles=internal_doubles)

    # Get list of amino acids to put at each degenerate codon
    amino_acids = _create_mut_aa_list(degen_codon=degen_codon)
    
    # Go through each library combo
    lib_clone_dict = {}
    desc = "{}".format("generating library sequences")
    for lib in tqdm(libraries_to_build,desc=desc,ncols=800):
    
        # List to hold clones
        library_clones = []
        
        # Figure out sites within each sub-library
        for_product = []
        for block in lib:
            sites = [i for i, s in enumerate(lib_id) if s == block]
            for_product.append(sites)
    
        # Make a list of combos to make at a site level (5,12), etc.
        combos_to_make = []
        for c in itertools.product(*for_product):
            
            # Sort by amino acid to avoid duplicates like 5,12 vs. 12,5
            combo = list(c)
            combo.sort()

            # Remove things like 5,5 or 12,12
            if len(combo) != len(set(combo)):
                continue
    
            combos_to_make.append(tuple(combo))
    
        # Remove duplicated combos (e.g. 5,12 coming from both the 5,12 and 12,5
        # combos
        combos_to_make = list(set(combos_to_make))
        combos_to_make.sort()
    
        # Now build out the site-level combinations into all possible amino acid
        # substitution codons. 

        # Go through each site combination (5,12), (6,12), (7,12), (5,13), ...
        for combo in combos_to_make:
    
            # Create list of amino acid mutations for each site combination. 
            # site_mutations will look like
            # [['R5A','R5C','R5D'...],['L12A','L12C','L12D',...]]
            site_mutations = []
            for site in combo:
    
                site_mut_list = []
                
                wt = aa_sequence[site]
                position = site + seq_starts_at
                for aa in amino_acids:
                    mutation = f"{wt}{position}{aa}"
                    site_mut_list.append(mutation)
    
                site_mutations.append(site_mut_list)
    
            # Generate all possible combinations of the site_mutations. So, 
            # make ["R5A","L12A"], ["R5A","L12C"] ... 
            for clone_mutants in itertools.product(*site_mutations):

                to_place = []
                for m in clone_mutants:

                    # Do not add something with same wildtype and mutant amino
                    # acid (e.g. R5R, L12L, etc.)
                    if m[0] == m[-1]:
                        continue
                    else:
                        to_place.append(m)

                if len(to_place) == 0:
                    clone_name = "wt"
                else:
                    clone_name = "/".join(to_place)

                library_clones.append(clone_name)

        lib_clone_dict[lib] = library_clones

    all_genotypes = []
    for lib in lib_clone_dict:
        all_genotypes.extend(lib_clone_dict[lib])

    # Now build a DataFrame with all genotypes
    genotype_df = _build_genotype_df(all_genotypes)

    return lib_clone_dict, genotype_df

