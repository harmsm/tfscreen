import tfscreen

from tfscreen.simulate import (
    library_prediction,
    selection_experiment
)

from typing import Any, Dict, Union
from pathlib import Path
import os

def _setup_file_output(output_dir,
                       output_prefix):

    if output_dir is None:
        return None
    
    if not isinstance(output_prefix,str):
        err = "if output_dir is specified, output_prefix must be a string\n"
        raise ValueError(err)
    
    roots = ["library","phenotype","genotype_ddG","sample","counts"]
    files = [f"{output_prefix}{r}.csv" for r in roots]
    files_with_path = [os.path.join(output_dir,f) for f in files]
    file_dict = dict([(roots[i],files_with_path[i]) for i in range(len(roots))])

    # Output directory exists
    if os.path.exists(output_dir):

        # But is not a directory
        if not os.path.isdir(output_dir):
            err = f"output_dir '{output_dir}' exists and is not a directory\n"
            raise FileExistsError(err)
        
    # Does not exist -- make it
    else:
        os.makedirs(output_dir)
    
    # Look for files that are already there and die if they are
    found_files = [f for f in files_with_path if os.path.exists(f)]
    if len(found_files) > 0:
        err = f"output files already exist: {','.join(found_files)}\n"
        raise ValueError(err)
        
    return file_dict
        

def run_simulation(cf: Union[Dict[str, Any], str, Path],
                   output_dir: Union[Dict[str, Any], str, Path],
                   output_prefix: str="tfscreen_",
                   override_keys: dict=None):
    """
    Simulate a full transcription factor selection and growth experiment, 

    This includes building the library, predicting phenotypes using a thermodynamic
    model, simulating growth, and then doing high-throughput sequencing. This 
    runs `library_prediction` and `selection_experiment` back-to-back. It also
    writes out files with the results in addition to returning the dataframes.

    Arguments
    ---------
    cf : dict or str or pathlib.Path
        The configuration for the simulation. Can be a dictionary or a path
        to a YAML file containing the configuration parameters.
    output_dir : None or str or pathlib.Path
        The output directory to write to. If None, do not write
    output_prefix : str
        put this prefix in front of all output files. default = "tfscreen_"
    override_keys : dict, default=None
        after reading the configuration file, replace keys in the configuration
        with the key/value pairs in override keys. No error checking is done 
        on these keys; the user is responsible for checking their sanity. 
    """
    
    # -------------------------------------------------------------------------
    # Read inputs and set up simulation

    cf = tfscreen.util.read_yaml(cf,override_keys=override_keys)
    if cf is None:
        err = "Aborting simulation due to configuration error."
        raise RuntimeError(err)
    
    # Decide if we are going to write out files, make output directory, and make
    # sure that we are not going to write over anything
    file_dict = _setup_file_output(output_dir,output_prefix)
    
    # Build library and predict its phenotypes
    library_df, phenotype_df, genotype_ddG_df = library_prediction(cf)

    # Perform selection experiment
    sample_df, counts_df = selection_experiment(cf,library_df,phenotype_df)

    # Prepare outputs
    out_dict = {"library":library_df,
                "phenotype":phenotype_df,
                "genotype_ddG":genotype_ddG_df,
                "sample":sample_df,
                "counts":counts_df}
    
    # If file_dict is specified, write out the results as files
    if file_dict is not None:
        for root in out_dict:
            out_dict[root].to_csv(file_dict[root],index=False)
    
    # Return outputs
    return out_dict


    

