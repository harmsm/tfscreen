"""
Generate libraries based on degenerate codon information.
"""
from data import degen_base_specifier
from data import codon_to_aa

import string
import itertools
import re

def _create_mut_aa_list(degen_codon):
    """
    Get a list of the amino acids encoded by a possibly degenerate codon.
    """

    amino_acids = []
    for b0 in list(degen_base_specifier[degen_codon[0]]):
        for b1 in list(degen_base_specifier[degen_codon[1]]):
            for b2 in list(degen_base_specifier[degen_codon[2]]):
                aa = codon_to_aa[f"{b0}{b1}{b2}"]
                amino_acids.append(aa)

    return amino_acids


def _get_libs_to_build(mutated_sites,
                       max_num_combos=2,
                       internal_doubles=False):
    """
    Build a list of all combos of up to max_num_combos sub libraries from a 
    string of mutated sites.

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
    Taken an amino acid sequence and description of sites to mutate and generate
    a list of libraries. Libraries are spit out at an amino acid mutation level
    and correspond to all clones. If two sub-libraries both encode S7G by
    itself, this will appear twice. The idea is to have every clone that will be
    present, even if this means repeating specific amino acid sequence.
    
    Parameters
    ----------
    aa_sequence : str
        string containing amino acid sequence as uppercase letters. whitespace
        and line-breaks are allowed. For example: MASTRKEQRVTLFGMNQR...
    mutated_sites : str
        string containing amino acid sequence as uppercase letters with mutated
        sites replaced with library identifier characters. whitespace
        and line-breaks are allowed. For example: MAST111QRVT222MNQR... This
        specifies there are two sublibraries "1" and "2" that replace the amino
        codon for that amino acid with 'degen_codon'. A library id can be 
        non-uppercase-letter character. 
    seq_starts_at : int, default = 1
        number of amino acid corresponding to first position in aa_seq.
    max_num_combos : int, default = 2
        maximum number of combinations to make. 1 means do each library only 
        individual; 2 means make pairwise combos; 3 means make three-way, etc. 
    internal_doubles : bool, default=False
        if True, make double mutants within a library block. for the 
        mutated_sites example above, this would make all combos within 1 and
        within 2, in addition to 1, 2, and (1,2). 
    degen_codon : str, default="nnt"
        use this codon at each site that is mutated


    Returns
    -------
    lib_clone_dict : dict
        dictionary keying library combinations to a list of clones. For the 
        examples above, this would have three keys ("1",), ("2",), ("1","2",). 
        Values are lists like [['R5A'],['R5C'],...]. Internal lists can have 
        multiple mutations ['R5A','L12A'] or none ([] ; wildtype).
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
    for lib in libraries_to_build:
    
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
                library_clones.append(to_place)
                
        lib_clone_dict[lib] = library_clones
                
    return lib_clone_dict
        
