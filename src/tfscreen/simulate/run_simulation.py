
from tfscreen.simulate import build_sample_dataframes
from tfscreen.simulate import generate_libraries
from tfscreen.simulate import generate_phenotypes
from tfscreen.simulate import transform_and_mix
from tfscreen.simulate import initialize_population
from tfscreen.simulate import simulate_growth
from tfscreen.simulate import sequence_samples
from tfscreen.simulate import load_simulation_config

from tfscreen.calibration import read_calibration

from tqdm.auto import tqdm

import os

def run_simulation(yaml_file: str,
                   output_prefix: str=None,
                   output_path: str=".",
                   override_keys: dict=None):
    """
    Simulate a full transcription factor selection and growth experiment.

    Arguments
    ---------
    yaml_file : str
        Path to the YAML configuration file.
    output_prefix : str, default=None
        output files will have output_prefix appended to front. if not specified,
        use 'replicate' from the yaml_file
    output_path : str, default="."
        write output file to this path
    override_keys : dict, default=None
        after reading the configuration file, replace keys in the configuration
        with the key/value pairs in override keys. No error checking is done 
        on these keys; the user is responsible for checking their sanity. 
    """
    
    # -------------------------------------------------------------------------
    # Read inputs and set up simulation

    desc = "{}".format(f"loading configuration '{yaml_file}'")
    with tqdm(total=1,desc=desc,ncols=800) as pbar:
        pbar.update()

    cf = load_simulation_config(yaml_file)
    if cf is None:
        print("Aborting simulation due to configuration error.")
        return

    # Programmatically replace keys from the configuration
    if override_keys is not None:
        for k in override_keys:
            if k not in cf:
                print(f"Warning: override_keys has a key '{k}' that was not in configuration.",flush=True)
            cf[k] = override_keys[k]

    if output_prefix is None:
        output_prefix = cf["replicate"]
    
    if os.path.exists(output_path):
        
        if not os.path.isdir(output_path):
            err = f"{output_path} already exists and is not a directory.\n"
            raise FileExistsError(err)

    else:
        os.mkdir(output_path)
    
    # -------------------------------------------------------------------------
    # Build library
        
    libraries, genotype_df = generate_libraries(
        aa_sequence=cf['aa_sequence'],
        mutated_sites=cf['mutated_sites'],
        seq_starts_at=cf['seq_starts_at'],
        max_num_combos=cf['max_num_combos'],
        internal_doubles=cf['internal_doubles'],
        degen_codon=cf['degen_codon']
    )

    sample_df, sample_df_with_time = build_sample_dataframes(
        cf['condition_blocks'],
        replicate=cf['replicate']
    )

    # -------------------------------------------------------------------------
    # Calculate phenotypes
        
    # Get calibrated description of wildtype growth
    calibration_dict = read_calibration(cf['calibration_file'])

    phenotype_df, genotype_df = generate_phenotypes(
        genotype_df=genotype_df,
        sample_df=sample_df,
        ensemble_spreadsheet=cf['ensemble_spreadsheet'],
        ddG_spreadsheet=cf['ddG_spreadsheet'],
        calibration_dict=calibration_dict,
        scale_obs_by=cf['scale_obs_by'],
        mut_growth_rate_std=cf['mut_growth_rate_std'],
        T=cf['T'],
        R=cf['R']
    )

    phenotype_df.to_csv(os.path.join(output_path,
                                        f"{output_prefix}-phenotype_df.csv"),index=False)
    assert False

    # -------------------------------------------------------------------------
    # Sample from library and grow out
    
    # Sample from the main library
    input_library = transform_and_mix(
        libraries=libraries,
        transform_sizes=cf['transform_sizes'],
        library_mixture=cf['library_mixture'],
        skew_sigma=cf['skew_sigma'],
        lambda_value=cf['lambda_value'],
        max_num_plasmids=cf['max_num_plasmids']
    )

    # Create initial populations and growth rates
    init_output = initialize_population(
        input_library=input_library,
        phenotype_df=phenotype_df,
        genotype_df=genotype_df,
        sample_df=sample_df,
        num_thawed_colonies=cf['num_thawed_colonies'],
        overnight_volume_in_mL=cf['overnight_volume_in_mL'],
        pre_iptg_cfu_mL=cf['pre_iptg_cfu_mL'],
        iptg_out_growth_time=cf['iptg_out_growth_time'],
        post_iptg_dilution_factor=cf['post_iptg_dilution_factor'],
        growth_rate_noise=cf['growth_rate_noise']
    )
    bacteria, ln_pop_array, bact_sample_k = init_output

    # Simulate the growth of each sample
    sample_pops = simulate_growth(
        ln_pop_array=ln_pop_array,
        bact_sample_k=bact_sample_k,
        sample_df=sample_df,
        sample_df_with_time=sample_df_with_time
    )

    # -------------------------------------------------------------------------
    # Sequence all libraries
        
    count_df, sample_df_with_time = sequence_samples(
        sample_pops=sample_pops,
        sample_df_with_time=sample_df_with_time,
        genotype_df=genotype_df,
        bacteria=bacteria,
        total_num_reads=cf['total_num_reads'],
        index_hop_freq=cf['index_hop_freq']
    )

    # -------------------------------------------------------------------------
    # Build final outputs and write as csv files
        
    cfu_dict = {}
    num_reads_dict = {}
    for c in tqdm(sample_df_with_time.index,desc="Assembling combined dataframe."):
        cfu = float(sample_df_with_time.loc[c, "cfu_per_mL"])
        num_reads = int(sample_df_with_time.loc[c, "num_reads"])
        cfu_dict[c] = cfu
        num_reads_dict[c] = num_reads

    count_df["total_cfu_mL_at_time"] = count_df["timepoint"].map(cfu_dict)
    count_df["total_counts_at_time"] = count_df["timepoint"].map(num_reads_dict)
    
    desc = "{}".format("writing file")
    with tqdm(total=3,desc=desc,ncols=800) as pbar:

        phenotype_df.to_csv(os.path.join(output_path,
                                        f"{output_prefix}-phenotype_df.csv"),index=False)
        pbar.update()

        sample_df.to_csv(os.path.join(output_path,f"{output_prefix}-sample_df.csv"))
        pbar.update()

        count_df.to_csv(os.path.join(output_path,f"{output_prefix}-combined_df.csv"),index=False)
        pbar.update()

    print(f"\n Simulation '{output_prefix}' complete.")