"""
Functions for sequencing simulated populations and collating results in tfscreen.
"""

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from tfscreen import get_cfu

def _sequence_single_pop(ln_pop_array,
                         input_library,
                         all_genotypes,
                         num_reads=50e6):
    """
    Simulate sequencing a population.

    Parameters
    ----------
    ln_pop_array : numpy.ndarray
        Log population array for each clone.
    input_library : numpy.ndarray
        Array holding bacterial clones in the population.
    all_genotypes : list
        List of all genotypes with full genotype information.
    num_reads : int, optional
        Number of sequencing reads to simulate (default: 50e6).

    Returns
    -------
    results : dict
        Dictionary mapping genotype string to read count.
    """

    # convert number of reads to integer
    num_reads = int(np.round(num_reads,0))

    # calculate sampling weight for all samples
    p = np.exp(ln_pop_array)
    p = p/np.sum(p)

    # indexes to select from
    select_from = np.arange(input_library.shape[0],
                            dtype=int)

    # indexes to grab
    samples = np.random.choice(select_from,num_reads,replace=True,p=p)
    samples, counts = np.unique(samples, return_counts=True)

    # Decide if this is a single plasmid
    single_plasmid = True
    if len(input_library.shape) > 1:
        single_plasmid = False
    
    results = {}
    for i, s in enumerate(samples):

        if single_plasmid:
            bacterium = input_library[s]
        else:
            bacterium = np.random.choice([v for v in input_library[s] if v > -1],1)[0]

        mut = "/".join(all_genotypes[bacterium]["clone"])
        if mut == "":
            mut = "wt"
            
        if mut not in results:
            results[mut] = 0

        results[mut] += counts[i]

    return results


def _sequence_everything(pops_vs_time,
                         iptg_concs,
                         sample_times,
                         input_library,
                         all_genotypes,
                         num_reads_per_condition,
                         replicate=1):
    """
    Simulate sequencing for all conditions and time points.

    Parameters
    ----------
    pops_vs_time : dict
        Dictionary keying selector to results. Values are arrays of ln_pop.
    iptg_concs : numpy.ndarray
        IPTG concentrations in mM.
    sample_times : numpy.ndarray
        Times where measurements were made (minutes).
    input_library : numpy.ndarray
        Array holding bacterial clones in the population.
    all_genotypes : list
        List of all genotypes with full genotype information.
    num_reads_per_condition : int
        Number of sequencing reads to simulate per condition.
    replicate : int, optional
        Replicate number for the sequencing (default: 1).

    Returns
    -------
    sequencing_results : list
        List of tuples (selector, time, iptg, sample dict).
    condition_df : pandas.DataFrame
        DataFrame with total counts and cfu/mL for each condition (selector,
        iptg, time).
    """
    sequencing_results = []
    
    cond_out_dict = {"condition_id":[], 
                     "replicate":[],
                     "marker":[],
                     "selection":[],
                     "iptg":[],
                     "time":[],
                     "cfu_per_mL":[],
                     "num_reads":[]}  
    
    print("sequencing samples of each condition",flush=True)
    num_conditions = len(pops_vs_time)*len(sample_times)*len(iptg_concs)
 
    with tqdm(total=num_conditions) as pbar:

        for sel in pops_vs_time:
            marker_name = sel[4:]
            for i, t in enumerate(sample_times):
                at_time_t = pops_vs_time[sel][i]
                for j, iptg in enumerate(iptg_concs):
                    at_iptg = at_time_t[:,j]
    
                    condition_id = f"{marker_name}-1-{iptg}-{t}-1"
                    cond_out_dict["condition_id"].append(condition_id)
                    cond_out_dict["replicate"].append(replicate)
                    cond_out_dict["marker"].append(marker_name)
                    cond_out_dict["selection"].append(True)
                    cond_out_dict["iptg"].append(iptg)
                    cond_out_dict["time"].append(t)
                    cond_out_dict["cfu_per_mL"].append(get_cfu(at_iptg))
                    cond_out_dict["num_reads"].append(num_reads_per_condition)
    
                    at_iptg_samples = _sequence_single_pop(at_iptg,
                                                           input_library=input_library,
                                                           all_genotypes=all_genotypes,
                                                           num_reads=num_reads_per_condition)
    


                    sequencing_results.append((marker_name,1,iptg,t,at_iptg_samples))
    
                                
                    pbar.update(1)
                
    condition_df = pd.DataFrame(cond_out_dict)
    condition_df = condition_df.sort_values(by=["replicate","marker","selection","iptg","time"],axis=0)
    condition_df = condition_df.reset_index(drop=True)

    return sequencing_results, condition_df


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
    
    unique_genotypes = []
    for g in all_genotypes:
        genotype = g["clone"]
        unique_genotypes.append("/".join(genotype))
            
    unique_genotypes = set(unique_genotypes)
    
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
            if mut[0] == "":
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

    return df

def _get_main_results(sequencing_results,
                      genotype_df,
                      cfu_df):
    """
    Extract main results from sequencing and cfu data.

    Parameters
    ----------
    sequencing_results : list
        List of tuples (selector, time, iptg, sample dict).
    genotype_df : pandas.DataFrame
        DataFrame with genotype information.
    cfu_df : pandas.DataFrame
        DataFrame with total cfu/mL for each condition and time point.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with genotype, selector, iptg, time, count, frequency, and
        cfu_per_mL.
    """
    
    cfu_dict = {}
    for s in np.unique(cfu_df["selector"]):
        cfu_dict[s] = {}
        for iptg in np.unique(cfu_df["iptg"]):
            cfu_dict[s][iptg] = {}
            for t in np.unique(cfu_df["time"]):
                cfu_dict[s][iptg][t] = {}
    
    for idx in cfu_df.index:
        row = cfu_df.loc[idx,:]
        cfu_dict[row["selector"]][row["iptg"]][row["time"]] = float(row["cfu_per_mL"])
    
    
    out_dict = {"genotype":[],
                "selector":[],
                "iptg":[],
                "time":[],
                "count":[],
                "freq":[],
                "cfu_per_mL":[]}
    
    num_genotypes = len(genotype_df["genotype"])
    
    print("extracting sequence counts",flush=True)
    for r in tqdm(sequencing_results):
    
        count_vector = []
        for g in genotype_df["genotype"]:
            
            if g in r[-1]:
                count_vector.append(r[-1][g])
            else:
                count_vector.append(0)
    
        count_vector = np.array(count_vector)
        freq_vector = count_vector/np.sum(count_vector)
        cfu_vector = freq_vector*cfu_dict[r[0]][r[2]][r[1]]
    
        out_dict["genotype"].extend(genotype_df["genotype"])
        out_dict["selector"].extend([r[0] for _ in range(num_genotypes)])
        out_dict["iptg"].extend([r[2] for _ in range(num_genotypes)])
        out_dict["time"].extend([r[1] for _ in range(num_genotypes)])
        out_dict["count"].extend(count_vector)
        out_dict["freq"].extend(freq_vector)
        out_dict["cfu_per_mL"].extend(cfu_vector)
        
    df = pd.DataFrame(out_dict)
    df["genotype"] = pd.Categorical(df["genotype"],genotype_df["genotype"])
    df = df.sort_values(["genotype","selector","iptg","time"])
    df = df.reset_index(drop=True)
    
    return df


def sequence_and_collate(pops_vs_time,
                         iptg_concs,
                         sample_times,
                         input_library,
                         all_genotypes,
                         num_reads_per_condition=50e6):
    """
    Generate final dataframes corresponding to primary outputs from the 
    experimental screen. This will be counts for each genotype at each
    condition (selector, iptg concentration, time), a dataframe with genotype
    information, and a dataframe with the total cfu/mL and number of reads for
    each condition.
    
    Parameters
    ----------
    pops_vs_time : dict
        Dictionary keying selector to results. Values are num_clones x 
        num_iptg x num_time_points arrays of ln_pop, generated by
        growth_with_selection.
    iptg_concs : numpy.ndarray
        IPTG concentrations in mM.
    sample_times : numpy.ndarray
        Times where measurements were made (minutes).
    input_library : numpy.ndarray
        Array holding bacterial clones in the population.
    all_genotypes : list
        List of all genotypes with full genotype information.
    num_reads_per_condition : int, optional
        Number of sequencing reads to simulate per condition (default: 50e6).

    Returns
    -------
    df : pandas.DataFrame
        All results. Has columns for genotype, selector, iptg, time, counts,
        frequency (in condition population), and cfu_per_mL (number of cells in
        condition population).
    genotype_df : pandas.DataFrame
        DataFrame with more detailed information about each genotype (sites,
        etc.).
    cfu_df : pandas.DataFrame
        The total cfu/mL for each condition at each time point.
    """

    genotype_df = _build_genotype_df(all_genotypes)

    sequencing_results, condition_df = _sequence_everything(pops_vs_time,
                                                            iptg_concs,
                                                            sample_times,
                                                            input_library,
                                                            all_genotypes,
                                                            num_reads_per_condition)

    df = _get_main_results(sequencing_results,
                           genotype_df,
                           condition_df)

    return df, genotype_df, condition_df