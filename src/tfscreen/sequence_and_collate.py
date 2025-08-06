
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from cell_growth_moves import get_cfu

def _sequence_single_pop(ln_pop_array,
                         input_library,
                         all_genotypes,
                         num_reads=50e6):
    """
    Simulate sequencing a population.
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

    
    # Decide if this is a single plasmid
    single_plasmid = True
    if len(input_library.shape) > 1:
        single_plasmid = False
    
    results = {}
    for s in samples:

        if single_plasmid:
            bacterium = input_library[s]
        else:
            bacterium = np.random.choice([v for v in input_library[s] if v > -1],1)[0]

        mut = "/".join(all_genotypes[bacterium]["clone"])
        if mut == "":
            mut = "wt"
            
        if mut not in results:
            results[mut] = 0
        results[mut] += 1

    return results


def _sequence_everything(pops_vs_time,
                         iptg_concs,
                         sample_times,
                         input_library,
                         all_genotypes,
                         num_reads_per_condition):
    
    sequencing_results = []
    
    cfu_out_dict = {"selector":[],
                    "iptg":[],
                    "time":[],
                    "cfu_per_mL":[]}
    
    print("sequencing samples of each condition",flush=True)
    num_conditions = len(pops_vs_time)*len(sample_times)*len(iptg_concs)
 
    with tqdm(total=num_conditions) as pbar:

        for sel in pops_vs_time:
            selection_name = sel[4:]
            for i, t in enumerate(sample_times):
                at_time_t = pops_vs_time[sel][i]
                for j, iptg in enumerate(iptg_concs):
                    at_iptg = at_time_t[:,j]
    
                    cfu_out_dict["selector"].append(selection_name)
                    cfu_out_dict["iptg"].append(iptg)
                    cfu_out_dict["time"].append(t)
                    cfu_out_dict["cfu_per_mL"].append(get_cfu(at_iptg))
    
                    at_iptg_samples = _sequence_single_pop(at_iptg,
                                                           input_library=input_library,
                                                           all_genotypes=all_genotypes,
                                                           num_reads=num_reads_per_condition)
    
                    sequencing_results.append((selection_name,t,iptg,at_iptg_samples))
    
                    
                                
                    pbar.update(1)
                
    cfu_df = pd.DataFrame(cfu_out_dict)
    cfu_df = cfu_df.sort_values(by=["selector","iptg","time"],axis=0)
    cfu_df = cfu_df.reset_index(drop=True)

    return sequencing_results, cfu_df


def _build_genotype_df(all_genotypes):
    
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
    Take the population over time arrays, sequence each pool, and generate 
    human-readable data frames with all results.
    
    Parameters
    ----------
    pops_vs_time : dict
        dictionary keying selector to results. values are
        num_clones x num_iptg x num_time_points arrays of ln_pop. generated by
        growth_with_selection
    iptg_concs : numpy.ndarray
        iptg concentrations in mM
    sample_times : numpy.ndarray
        times where measurements were made (minutes)
    input_library : numpy.ndarray
        array holding bacterial clones in the population
    all_genotypes : list
        list of all genotypes with full genotype information
    num_reads_per_condition : int, default=50e6
        number of sequencing reads to simulate per condition

    Returns
    -------
    df : pandas.DataFrame
        all results. has columns for genotype, selector, iptg, time, counts,
        frequency (in condition population), and cfu_per_mL (number of cells in
        condition population)
    genotype_df : pandas.DataFrame
        dataframe with more detailed information about each genotype (sites, etc.)
    cfu_df : pandas.DataFrame
        the total cfu/mL for each condition at each time point
    """

    genotype_df = _build_genotype_df(all_genotypes)

    sequencing_results, cfu_df = _sequence_everything(pops_vs_time,
                                                      iptg_concs,
                                                      sample_times,
                                                      input_library,
                                                      all_genotypes,
                                                      num_reads_per_condition)

    df = _get_main_results(sequencing_results,
                           genotype_df,
                           cfu_df)

    return df, genotype_df, cfu_df