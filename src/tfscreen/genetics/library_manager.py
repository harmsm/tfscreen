
from tfscreen.util import (
    read_yaml,
    check_number
)
from tfscreen.data import (
    CODON_TO_AA,
    DEGEN_BASE_SPECIFIER,
)

from itertools import (
    groupby,
    product
)

from typing import Iterable, Set, Dict, Union, Tuple, List


def _check_char(some_str: str,
                name: str,
                allowed_chars: Iterable[str]) -> None:
    """
    Checks if all characters in a string are from an allowed set.

    Parameters
    ----------
    some_str : str
        The string to validate.
    name : str
        The name of the variable being checked, used for clear error messages.
    allowed_chars : Iterable[str]
        An iterable of characters that are permitted in `some_str`.

    Raises
    ------
    ValueError
        If `some_str` contains one or more characters that are not in
        `allowed_chars`.

    """
    str_set = set(some_str)
    allowed_set = set(allowed_chars)

    if not str_set.issubset(allowed_set):
        # Sort for a deterministic error message, which is good for testing
        unrecognized_chars = "".join(sorted(list(str_set - allowed_set)))
        raise ValueError(
            f"Not all characters in {name} were recognized. Characters not "
            f"recognized: '{unrecognized_chars}'"
        )

def _check_contiguous_lib_blocks(s: str) -> None:
    """
    Checks if all non-'.' characters in a string form contiguous blocks.

    This function validates that a sub-library (e.g., '1') is defined as a
    single, unbroken block ('..111..') and not split into multiple
    segments ('..1.11..'). The '.' character is treated as a neutral filler.

    Parameters
    ----------
    s : str
        The input string to check.

    Raises
    ------
    ValueError
        If a character block (other than '.') is not contiguous.

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
                   libraries_seen: Set[str]) -> None:
    """
    Validates the format of a library combination key.

    A valid key must be in the format 'single-x' or 'double-x-y', where
    'x' and 'y' are characters representing sub-libraries that have been
    defined in the main configuration.

    Parameters
    ----------
    lib_key : str
        The library combination key string to validate (e.g., 'single-1',
        'double-1-2').
    libraries_seen : set[str]
        A set of all valid sub-library characters found in the
        'sub_libraries' configuration string.

    Raises
    ------
    ValueError
        If the key has an invalid format, specifies an unknown command, has
        the wrong number of parts, or references a sub-library that does
        not exist.

    """
    parts = lib_key.strip().split("-")
    command = parts[0]
    
    # Generic help text for error messages
    help_text = (
        "Keys must be 'single-x' or 'double-x-y', where x and y are "
        "defined sub-libraries."
    )

    if command == "single":
        if len(parts) != 2:
            raise ValueError(
                f"Invalid key '{lib_key}'. 'single' keys must have one "
                f"sub-library part. {help_text}"
            )
    elif command == "double":
        if len(parts) != 3:
            raise ValueError(
                f"Invalid key '{lib_key}'. 'double' keys must have two "
                f"sub-library parts. {help_text}"
            )
    else:
        raise ValueError(
            f"Invalid command '{command}' in key '{lib_key}'. "
            f"Command must be 'single' or 'double'."
        )

    # Check if all specified sub-libraries are valid
    sub_libs = parts[1:]
    for lib_id in sub_libs:
        if lib_id not in libraries_seen:
            raise ValueError(
                f"Unrecognized sub-library '{lib_id}' in key '{lib_key}'. "
                f"Valid libraries are: {sorted(list(libraries_seen))}. "
                f"{help_text}"
            )
      
class LibraryManager:

    # Check for missing keys in config
    REQUIRED_KEYS = ["reading_frame",
                     "first_amplicon_residue",
                     "wt_seq",
                     "degen_sites",
                     "sub_libraries",
                     "library_combos"]

    def __init__(self,run_config):

        # -- Build sets of base names to check input --
        self.standard_bases = set(list("".join(CODON_TO_AA.keys())))
        self.degen_bases = set(DEGEN_BASE_SPECIFIER.keys())
        self.standard_plus_dot = self.standard_bases.union({"."})
        self.degen_plus_dot = self.degen_bases.union({"."})

        self._parse_and_validate(run_config)
        self._prepare_blocks()
        self._prepare_indexes()
     
    def _parse_and_validate(self,
                            run_config: Union[str, Dict]) -> None:
        """
        Parses and validates the run configuration.

        This method reads a configuration from a file path or dictionary,
        validates all fields for correctness and consistency, and populates
        the instance with derived attributes.

        Parameters
        ----------
        run_config : str or dict
            Either a path to a YAML configuration file or a dictionary
            containing the configuration parameters.

        Raises
        ------
        ValueError
            If the configuration is invalid due to missing keys, incorrect
            data types, inconsistent values (e.g., mismatched sequence
            lengths), or violations of the library definition rules.

        Notes
        -----
        This method has the side effect of setting the following attributes
        on the class instance upon successful validation:
        - `reading_frame`
        - `first_amplicon_residue`
        - `wt_seq`
        - `degen_sites`
        - `sub_libraries`
        - `libraries_seen`
        - `library_combos`
        - `expected_length`
        - `aa_seq`
        - `degen_seq`
        - `run_config`
        """
    
        # Read run config (yaml or pass through if already a dict)
        run_config = read_yaml(run_config)
        if run_config is None:
            raise ValueError(
                f"could not read '{run_config}'."
            )
        
        # Store the run_config as an attribute so we can access its values
        self.run_config = run_config

        missing = [k for k in self.REQUIRED_KEYS if k not in run_config]
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


        # -- Validate the sequence/library specification --
        
        # Load wildtype seq
        wt_seq = str(run_config["wt_seq"]).strip()
        _check_char(wt_seq,"wt_seq",self.standard_bases)
        
        # Load degenerate sites
        degen_sites = str(run_config["degen_sites"]).strip()
        _check_char(degen_sites,"degen_sites",self.degen_plus_dot)
        
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
            if degen_sites[i] not in self.standard_plus_dot:
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

    def _prepare_blocks(self) -> None:
        """
        Processes sequence definitions to create combinatorial blocks.

        This method iterates through the `self.sub_libraries` string and
        constructs three parallel lists that represent the entire sequence as a
        series of blocks. These blocks are the fundamental units used for
        combinatorially generating mutant libraries.

        For wild-type regions (marked with '.'), each base is treated as a
        separate, single-character block. For sub-library regions (e.g., '111'),
        the entire contiguous region is processed at once by a helper method
        to generate blocks corresponding to codons.

        Notes
        -----
        This method does not return any value but sets the following instance
        attributes:

        self.wt_blocks : list[list[str]]
            A list where each inner list contains the wild-type sequence for a
            single block (e.g., `['c']` for a base or `['gac']` for a codon).
        self.mut_blocks : list[list[str]]
            A parallel list to `wt_blocks`. For wild-type blocks, it is
            identical. For degenerate codon blocks, the inner list contains all
            possible codon sequences (e.g., `['gct', 'gcc', 'gca', 'gcg']`).
        self.lib_lookup : list[str]
            A parallel list that maps each block index to its sub-library
            identifier (e.g., '.', '1', '2').
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


    def _prepare_indiv_lib_blocks(self, lib_to_get: str) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Generates wt and mut blocks for a single contiguous sub-library.

        This method isolates a specific sub-library region (e.g., all '1's),
        aligns it to the instance's reading frame, and processes it into
        combinatorial blocks. It handles out-of-frame bases by separating them
        into non-combinatorial "flank" blocks and ensures these flanks do not
        contain degenerate bases.

        Parameters
        ----------
        lib_to_get : str
            The character identifier for the sub-library to process (e.g., '1').

        Returns
        -------
        tuple[list[list[str]], list[list[str]]]
            A tuple containing two lists: (wt_blocks, mut_blocks).
            - wt_blocks: A list of wild-type sequence blocks for this region.
            - mut_blocks: A parallel list containing all possible sequences for
            each corresponding block. For flanks, this is identical to wt_blocks.

        Raises
        ------
        ValueError
            If any out-of-frame "flank" bases are found to be degenerate (i.e.,
            not standard 'a', 'c', 'g', or 't' bases).
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
            left_flank = ""
        else:
            left_flank = lib_seq[:offset]

        # Trim the left flank from the main sequence
        core_lib_seq = lib_seq[offset:]
        core_wt_seq = wt_seq[offset:]

        # Now, calculate trailing bases and slice them off
        num_trailing = len(core_lib_seq) % 3
        if num_trailing == 0:
            right_flank = ""
        else:
            right_flank = core_lib_seq[-num_trailing:]
            core_lib_seq = core_lib_seq[:-num_trailing]
            core_wt_seq = core_wt_seq[:-num_trailing]

        # Re-assign to original variable names for the rest of the function
        lib_seq = core_lib_seq
        wt_seq = core_wt_seq

        # Make sure the flanks don't have degenerate codons -- just standard bases
        # Use the existing self.standard_bases attribute
        for flank, name in [(left_flank, "left_flank"), (right_flank, "right_flank")]:
            try:
                _check_char(flank, name, self.standard_bases)
            except ValueError as e:
                raise ValueError(
                    f"Degenerate bases must be within codons within a sub-library. "
                    f"Sequence '{flank}' is out of the main reading frame "
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
        """
        Creates index and residue number lookups for library generation.

        This method builds two essential data structures needed by the library
        generation methods: a mapping from sub-library IDs to their block
        indices and a list of residue numbers for naming mutations.

        Notes
        -----
        This method does not return any value but sets the following instance
        attributes:

        self.indexers : dict[str, list[int]]
            A dictionary mapping each sub-library identifier (e.g., '1') to a
            list of the integer indices where that library's blocks appear in
            the main block lists (`wt_blocks`, `mut_blocks`).
        self.residues : list[str]
            A list of strings, where each string is the amino acid residue
            number corresponding to each block. This is offset by
            `self.first_amplicon_residue`.
        """

        self.indexers = {}
        for lib in self.libraries_seen:
            self.indexers[lib] = [i for i in range(len(self.lib_lookup))
                                  if self.lib_lookup[i] == lib]

        self.residues = [f"{i + self.first_amplicon_residue}"
                         for i in range(len(self.wt_blocks))]
        
    
    def _convert_to_aa(self, lib_seqs: List[str]) -> List[str]:
        """
        Translates DNA sequences into formatted amino acid mutation strings.

        For each DNA sequence provided, this method translates it to an amino
        acid sequence and compares it to the wild-type protein sequence. It
        then generates a standardized, slash-separated string describing the
        changes (e.g., "A42T/Q90R").

        Parameters
        ----------
        lib_seqs : list[str]
            A list of DNA sequences to be translated and compared.

        Returns
        -------
        list[str]
            A list of mutation strings. For sequences with no amino acid
            changes (including synonymous mutations), an empty string "" is
            returned.

        Notes
        -----
        This method relies on the following instance attributes having been
        previously set: `self.reading_frame`, `self.aa_seq` (wild-type), and
        `self.residues`.
        """
        
        aa_muts = []
        for lib_member in lib_seqs:
            seq = lib_member[self.reading_frame:]
            seq = seq[:(len(seq) - len(seq) % 3)]
            aa_seq = "".join([CODON_TO_AA[seq[c:(c+3)]] for c in range(0,len(seq),3)])
    
            wt_res_mut = zip(self.aa_seq,self.residues,aa_seq) 
            aa_mut_list = ["".join([wt,res,mut])
                           for wt,res,mut in wt_res_mut
                           if wt != mut]
            
            aa_muts.append("/".join(aa_mut_list))
            
        return aa_muts

    def _get_singles(self, target_lib: str) -> Tuple[List[str], List[str]]:
        """
        Generates all single mutants for a specific sub-library.

        This method iterates through each mutable block defined for the target
        sub-library. For each block, it generates all possible full-length DNA
        sequences where only that single block is mutated. The results from
        all blocks are combined into a single list.

        Parameters
        ----------
        target_lib : str
            The identifier of the sub-library (e.g., '1') for which to
            generate single mutants.

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple containing two lists: (lib_seqs, aa_muts).
            - lib_seqs: A list of all generated DNA sequences.
            - aa_muts: A parallel list of the corresponding formatted amino
            acid mutation strings.

        Notes
        -----
        This method relies on pre-computed attributes: `self.indexers`,
        `self.wt_blocks`, and `self.mut_blocks`. It calls the helper method
        `self._convert_to_aa` for the final translation step.
        The wild-type sequence will be present in the output list once for
        each mutable position in the sub-library.

        """

        indexer = self.indexers[target_lib]
        
        lib_seqs = []
        for idx in indexer:
            base_seq = self.wt_blocks[:]
            base_seq[idx] = self.mut_blocks[idx]
            lib_seqs.extend(["".join(seq) for seq in product(*base_seq)])
    
        aa_muts = self._convert_to_aa(lib_seqs)
    
        return lib_seqs, aa_muts

    
    def _get_intra_doubles(self, target_lib: str) -> Tuple[List[str], List[str]]:
        """
        Generates all double-mutant combinations within a single sub-library.

        This method iterates through all unique pairs of mutable blocks within the
        target sub-library. For each pair, it generates all possible full-length
        DNA sequences.

        Parameters
        ----------
        target_lib : str
            The identifier of the sub-library (e.g., '1') in which to
            generate double mutants.

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple containing two lists: (lib_seqs, aa_muts).
            - lib_seqs: A list of all generated DNA sequences.
            - aa_muts: A parallel list of the corresponding formatted amino
            acid mutation strings.

        Notes
        -----
        The generated set is inclusive. For each pair of mutated sites, the
        output will contain not only the double mutants but also the two
        corresponding single mutants and the wild-type sequence.
        This method relies on pre-computed attributes (`self.indexers`, etc.)
        and calls `self._convert_to_aa` for translation.

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
    
    def _get_inter_doubles(self, target_lib_1: str, target_lib_2: str) -> Tuple[List[str], List[str]]:
        """
        Generates all double-mutant combinations between two sub-libraries.

        This method iterates through all pairs of mutable blocks where one block
        is from the first sub-library and the other is from the second. For
        each pair, it generates all possible full-length DNA sequences.

        Parameters
        ----------
        target_lib_1 : str
            The identifier of the first sub-library (e.g., '1').
        target_lib_2 : str
            The identifier of the second sub-library (e.g., '2').

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple containing two lists: (lib_seqs, aa_muts).
            - lib_seqs: A list of all generated DNA sequences.
            - aa_muts: A parallel list of the corresponding formatted amino
            acid mutation strings.

        Notes
        -----
        The generated set is inclusive. For each pair of mutated sites, the
        output will contain not only the double mutants but also the two

        corresponding single mutants and the wild-type sequence.
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
 

    def get_libraries(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Generates all libraries specified in the run configuration.

        This is the main method to generate the libraries after the class
        has been initialized. It iterates through the `library_combos` list
        from the configuration and calls the appropriate internal methods to
        generate single, intra-library double, or inter-library double mutants.

        Returns
        -------
        tuple[dict, dict]
            A tuple of two dictionaries: (all_lib_seqs, all_aa_muts).
            - all_lib_seqs: A dictionary where keys are the library combo
            strings (e.g., "single-1") and values are lists of the
            corresponding generated DNA sequences.
            - all_aa_muts: A parallel dictionary where keys are the library
            combo strings and values are lists of the corresponding
            formatted amino acid mutation strings.
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

