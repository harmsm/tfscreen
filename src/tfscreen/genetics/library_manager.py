from tfscreen.simulate.generate_libraries import _expand_degen_codon

from tfscreen.util import (
    read_yaml,
    check_number
)
from tfscreen.data import (
    CODON_TO_AA,
    DEGEN_BASE_SPECIFIER,
)

import pandas as pd

from itertools import (
    groupby,
    product
)

def _check_char(some_str: str,
                name: str,
                allowed_chars: str) -> None:
    """
    Make sure that all characteris in some_str are within allowed_chars.
    """
    
    str_set = set(some_str)
    if not str_set.issubset(allowed_chars):
        raise ValueError(
            f"not all characters in {name} were recognized. Characters not "
            f"recognized: '{str_set - allowed_chars}'"
        )

def _check_contiguous_lib_blocks(s: str) -> None:
    """
    Checks if all non-'.' characters in a string form contiguous blocks.

    For example:
    - "..111..22.." is True (all '1's are together, all '2's are together)
    - "..11.1..22.." is False ('1's are separated by a '.')

    Parameters
    ----------
    s : str
        input string to check.

    Raises
    ------
    ValueError :
        raised if character blocks beside '.' do not form contiguous blocks. 
    """
    
    seen_chars = set()
    for char, _ in groupby(s):
        # Ignore the '.' characters
        if char == '.':
            continue
        
        # If we see a character that has already been seen in a previous block,
        # then its occurrences are not contiguous.
        if char in seen_chars:
            raise ValueError(
                f"Sub-library '{char}' is not in a contiguous block. Each sub-"
                f"library must (except filler '.') must be contiguous. For "
                f"example, ..111..22.. is allowed but ..11.1..22.. is not."
            )
        
        # Otherwise, add the character to our set of seen characters.
        seen_chars.add(char)
        
def _check_lib_key(lib_key: str,
                   libraries_seen: iter) -> None:
    """
    Check the formatting and sanity of lib_key entries.
    """

    generic_err = ["lib_keys should be formatted like 'single-x' or",
                   "'double-x-y' where 'x' and 'y' are library characters in",
                   "sub_libraries. 'single-x' specifies all degenerate codons",
                   "in x. 'double-x-x' specifies doubles between degenerate",
                   "codons in x. 'double-x-y' specifies doubles between",
                   "sub-libraries x and y but not internal doubles."]
    generic_err = " ".join(generic_err)

    # Split lib_key into fields
    columns = str(lib_key).strip().split("-")

    # Check for recognized first column
    if columns[0] not in ["single","double"]:
        raise ValueError(generic_err)

    # Single, not enough/too many entries
    if columns[0] == "single" and len(columns) != 2:
        raise ValueError(
            f"single lib_keys must have exactly one sublibrary specified. "
            f"lib_key '{lib_key}' specifies {len(columns)-1} ({columns[1:]}). "
            + generic_err
        )

    # Double, not enough/too many entries
    if columns[0] == "double" and len(columns) != 3:
        raise ValueError(
            f"double lib_keys must have exactly two sublibraries specified. "
            f"lib_key '{lib_key}' specifies {len(columns)-1} ({columns[1:]}). "
            + generic_err
        )

    # Check to make sure we recognize the sub-libraries specified
    for k in columns[1:]:
        if k not in libraries_seen:
            raise ValueError(
                f"sub-library '{k}' not recognized. It should be one of '{libraries_seen}'. "
                + generic_err
            )
      
class LibraryManager:

    def __init__(self,run_config):

        self.run_config = run_config
        self._parse_and_validate(run_config)
        self._prepare_blocks()
        self._prepare_indexes()
     
    def _parse_and_validate(self,run_config):
        """
        Load a run configuration file/dictionary. 
        """
    
        # Read run config (yaml or pass through if already a dict)
        run_config = read_yaml(run_config)
        if run_config is None:
            raise ValueError(
                f"could not read '{run_config}'."
            )
        
        # Check for missing keys in config
        required_keys = ["reading_frame",
                         "first_amplicon_residue",
                         "wt_seq",
                         "degen_sites",
                         "sub_libraries",
                         "expected_5p",
                         "expected_3p",
                         "library_combos"]
        
        missing = [k for k in required_keys if k not in run_config]
        if len(missing) > 0:
            raise ValueError(
                f"run_config is missing keys: {missing}"
            )
            
        # -- Read and check single-value inputs --
        reading_frame = check_number(run_config["reading_frame"],
                                     cast_type=int,
                                     max_allowed=2,
                                     inclusive_max=True,
                                     min_allowed=0,
                                     inclusive_min=True)
        self.reading_frame = reading_frame
        
        first_amplicon_residue = check_number(run_config["first_amplicon_residue"],
                                              cast_type=int)
        self.first_amplicon_residue = first_amplicon_residue
        
        # -- Build sets of base names to check input --
        standard_bases = set(list("".join(CODON_TO_AA.keys())))
        degen_bases = set(DEGEN_BASE_SPECIFIER.keys())
        standard_plus_dot = standard_bases.union({"."})
        degen_plus_dot = degen_bases.union({"."})
    
        # -- Validate the sequence/library specification --
        
        # Load wildtype seq
        wt_seq = str(run_config["wt_seq"]).strip()
        _check_char(wt_seq,"wt_eq",standard_bases)
        
        # Load degenerate sites
        degen_sites = str(run_config["degen_sites"]).strip()
        _check_char(degen_sites,"degen_sites",degen_plus_dot)
        
        # Load sub-libraries
        sub_libraries = str(run_config["sub_libraries"]).strip()
        _check_contiguous_lib_blocks(sub_libraries)
        libraries_seen = set(list(sub_libraries)) - {"."}
        
        # Now do some cross-validation between seqs
        if len(wt_seq) != len(degen_sites) or len(wt_seq) != len(sub_libraries):
            raise ValueError("wt_seq, degen_sites, and sub_libraries must all be the same length")
        
        # Go through each column and makes sure that degenerate bases are only in 
        # sub-libraries. Build the error message on the fly by putting "!" in every 
        # problem column. 
        status = []
        for i in range(len(list(wt_seq))):
    
            # degen_sites must be standard bases unless they are part of a sub_library
            if degen_sites[i] not in standard_plus_dot:
                if sub_libraries[i] == ".":
                    status.append("!")
                    continue
            status.append(" ")
    
        # Problem. Build error message and return. 
        if "!" in status:
            err = "".join(["Degenerate bases are only allowed within sub_libraries.",
                           "The problematic columns are indicated with '!' below:\n"])
            
            final_err = "\n".join([err,
                                   "".join(status),wt_seq,degen_sites,sub_libraries,
                                   "\n"])
            raise ValueError(final_err)
    
        self.wt_seq = wt_seq
        self.degen_sites = degen_sites
        self.sub_libraries = sub_libraries
        self.libraries_seen = libraries_seen
    
        # -- Deal with library specification  --
        
        # Load expected 5' flank
        expected_5p = str(run_config["expected_5p"])
        _check_char(expected_5p,"expected_5p",standard_bases)
        self.expected_5p = expected_5p
        
        # Load expected 3' flank
        expected_3p = str(run_config["expected_3p"])
        _check_char(expected_3p,"expected_3p",standard_bases)
        self.expected_3p = expected_3p
    
        # Check library combos
        if isinstance(run_config["library_combos"],str) or not hasattr(run_config["library_combos"],"__iter__"):
            raise ValueError(
                f"library_combos should be a list of strings of library combinations. "
                f"library_combos '{run_config["library_combos"]}' is not valid."
            )
        self.library_combos = []
        for lib_key in run_config["library_combos"]:
            _check_lib_key(lib_key,self.libraries_seen)
            self.library_combos.append(lib_key)
    
        # -- Build sequences useful for later -- 
    
        self.expected_length = len(wt_seq)
        
        # Translate amino acid sequence using appropriate reading frame
        aa_seq = []
        for i in range(reading_frame,len(wt_seq),3):
            codon = "".join(wt_seq[i:(i+3)])
            if len(codon) < 3: break
            aa_seq.append(CODON_TO_AA[codon])
        self.aa_seq = "".join(aa_seq)
    
        # Build complete degen seq without '.' by putting together with wt
        degen_seq = [degen_sites[i] if degen_sites[i] != '.' else wt_seq[i]
                 for i in range(len(degen_sites))]
        self.degen_seq = "".join(degen_seq)

    def _prepare_blocks(self):
        """
        Set up blocks for combinatorial library generation
        """
    
        wt_blocks = []
        mut_blocks = []
        lib_lookup = []
        
        current_sub_lib = None
        for i in range(len(self.sub_libraries)):
        
            sub_lib = self.sub_libraries[i]
            if sub_lib == ".":
                wt_blocks.append([self.wt_seq[i]])
                mut_blocks.append([self.wt_seq[i]])
                lib_lookup.append(".")
            else:
                if current_sub_lib == sub_lib:
                    continue
                    
                w, m = self._prepare_indiv_lib_blocks(sub_lib)
                wt_blocks.extend(w)
                mut_blocks.extend(m)
                lib_lookup.extend([sub_lib for _ in w])
                
                current_sub_lib = sub_lib
        
        self.wt_blocks = wt_blocks
        self.mut_blocks = mut_blocks
        self.lib_lookup = lib_lookup


    def _prepare_indiv_lib_blocks(self,lib_to_get):
        """
        Generate lists of possible sequences at different blocks to allow 
        combinatorial assembly of specific libraries via itertools.product().
        """
    
        # Get list indexes for the sub library
        indexes = [i for i in range(len(self.sub_libraries))
                   if self.sub_libraries[i] == lib_to_get]
        start_idx = indexes[0]
        end_idx = indexes[-1] + 1
    
        # Get the degenerate sites and wildtype sequences for this sub-library
        lib_seq = "".join(self.degen_sites[start_idx:end_idx])
        wt_seq = "".join(self.wt_seq[start_idx:end_idx])
    
        # Extract the region of the library that encodes the degenerate library,
        # as well as any flanks leftover after the extraction
        start_frame = start_idx % 3
        offset = (self.reading_frame - start_frame) % 3 
        if offset == 0:
            left_flank = []
            right_flank = []
        else:
            left_flank = lib_seq[:offset]
            num_trailing = (len(lib_seq) - offset) % 3
            right_flank = lib_seq[-num_trailing:]
            lib_seq = lib_seq[offset:(-num_trailing)]
            wt_seq = wt_seq[offset:(-num_trailing)]
    
        # Make sure the flanks don't have degenerate codons -- just standard bases
        standard_bases = set(list("".join(CODON_TO_AA.keys())))
        for flank, name in [(left_flank,"left_flank"),(right_flank,"right_flank")]:
            try:
                _check_char(flank,name,standard_bases)
            except ValueError as e:
                raise ValueError(
                    f"Degenerate bases must be within codons within a sub-library. "
                    f"Sequence '{flank}' is before the reading frame "
                    f"({self.reading_frame}) but has non-standard bases."
                ) from e
    
        # Build wt_blocks (which hold the wildtype version of each block) and 
        # mut_blocks (which old the mutant versions of each block)
        wt_blocks = []
        mut_blocks = []
    
        # If the left_flank exists, append it
        if len(left_flank) > 0:
            wt_blocks.append([left_flank])
            mut_blocks.append([left_flank])
    
        # Now go through each site in the library
        for i in range(0,len(lib_seq),3):
    
            # Record the wildtype codon
            wt_blocks.append([wt_seq[i:(i+3)]])
        
            # Expand codon into a list of possible codings
            codon = lib_seq[i:(i+3)]
            degen = [list(DEGEN_BASE_SPECIFIER[base]) for base in codon]
            all_codons = ["".join(seq) for seq in product(*degen)]
            mut_blocks.append(all_codons)
    
        # If the right_flank exists, append it
        if len(right_flank) > 0:
            wt_blocks.append([right_flank])
            mut_blocks.append([right_flank])
    
        return wt_blocks, mut_blocks

    def _prepare_indexes(self):

        self.indexers = {}
        for lib in self.libraries_seen:
            self.indexers[lib] = [i for i in range(len(self.lib_lookup))
                                  if self.lib_lookup[i] == lib]

        self.residues = [f"{i + self.first_amplicon_residue}"
                         for i in range(len(self.wt_blocks))]
        
    
    def _convert_to_aa(self,lib_seqs):
        """
        Convert all nucleic acid sequences in lib_seqs into amino acid mutation
        descriptions like A42T/Q90T, etc.
        """
        
        aa_seqs = []
        aa_muts = []
        for lib_member in lib_seqs:
            seq = lib_member[self.reading_frame:]
            seq = seq[:(len(seq) - len(seq) % 3)]
            aa_seq = "".join([CODON_TO_AA[seq[c:(c+3)]] for c in range(0,len(seq),3)])
            aa_seqs.append(aa_seq)
    
            wt_res_mut = zip(self.aa_seq,self.residues,aa_seq) 
            aa_mut_list = ["".join([wt,res,mut])
                           for wt,res,mut in wt_res_mut
                           if wt != mut]
            
            aa_muts.append("/".join(aa_mut_list))
            
        return aa_muts

    def _get_singles(self,target_lib):
        """
        Get all possible single mutants within a single sub-library.
        """

        indexer = self.indexers[target_lib]
        
        lib_seqs = []
        for idx in indexer:
            base_seq = self.wt_blocks[:]
            base_seq[idx] = self.mut_blocks[idx]
            lib_seqs.extend(["".join(seq) for seq in product(*base_seq)])
    
        aa_muts = self._convert_to_aa(lib_seqs)
    
        return lib_seqs, aa_muts

    
    def _get_intra_doubles(self,target_lib):
        """
        Get all possible double mutations within a single sub-library.
        """
        
        indexer = self.indexers[target_lib]
    
        # Loop over all possible unique pairs of sites
        lib_seqs = []
        for i in range(len(indexer)):
            base_seq = self.wt_blocks[:]
            base_seq[indexer[i]] = self.mut_blocks[indexer[i]]
            
            for j in range(i+1,len(indexer)):
                double_seq = base_seq[:]
                double_seq[indexer[j]] = self.mut_blocks[indexer[j]]
                lib_seqs.extend(["".join(seq) for seq in product(*double_seq)])
    
        aa_muts = self._convert_to_aa(lib_seqs)
    
        return lib_seqs, aa_muts
    
    def _get_inter_doubles(self,target_lib_1,target_lib_2):
        """
        Get all possible double mutations betweenn two sub-libraries.
        """
        
        indexer_1 = self.indexers[target_lib_1]
        indexer_2 = self.indexers[target_lib_2]
    
        lib_seqs = []
        for idx1 in indexer_1:
            base_seq = self.wt_blocks[:]
            base_seq[idx1] = self.mut_blocks[idx1]
            
            for idx2 in indexer_2:
                double_seq = base_seq[:]
                double_seq[idx2] = self.mut_blocks[idx2]
                lib_seqs.extend(["".join(seq) for seq in product(*double_seq)])
    
        aa_muts = self._convert_to_aa(lib_seqs)
    
        return lib_seqs, aa_muts
 
    def get_libraries(self):
        """
        Get all libraries encoded in a run configuration.
        """
    
        all_lib_seqs = {}
        all_aa_muts = {}
    
        # Go through every library combo (single-1, double-1-2, etc.)
        for k in self.library_combos:
            cols = k.split("-")
    
            # Singles
            if cols[0] == "single":
                lib_seqs, aa_muts = self._get_singles(cols[1])
                all_lib_seqs[k] = lib_seqs
                all_aa_muts[k] = aa_muts
    
            # Doubles
            if cols[0] == "double":
        
                if cols[1] == cols[2]:        
                    lib_seqs, aa_muts = self._get_intra_doubles(cols[1])
                else:
                    lib_seqs, aa_muts = self._get_inter_doubles(cols[1],cols[2])
                    
                all_lib_seqs[k] = lib_seqs
                all_aa_muts[k] = aa_muts
                
        return all_lib_seqs, all_aa_muts

