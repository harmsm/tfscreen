
import numpy as np

def build_design_matrix(marker,
                        select,
                        iptg,
                        theta=None,
                        K=None,
                        n=None,
                        log_iptg_offset=1e-6,
                        param_names=None):
    """
    Build a design matrix in a stereotyped way. 

    Parameters
    ----------
    marker : np.ndarray
        1D array of sample markers (the special value 'none' is ignored). 
    select : np.ndarray
        1D array of selection state for each sample 
    iptg : np.ndarray
        1D array of iptg concentration for each sample
    K : float
        binding coefficient for the operator occupancy Hill model
    n : float
        Hill coefficient for the operator occupancy Hill model
    log_iptg_offset : float
        add this to value to all iptg concentrations before taking the log 
        to fit the linear model
    param_names : np.ndarray or None, default=None
        if param_names is passed in, this is treated as the names of each 
        column in the design matrix
        
    Methods
    -------
    param_names : list
        list of parameter names
    X : np.ndarray
        design matrix (num_samples by num parameters array)
    """

    # Get unique marker and selection values and sort. Ignore "none" markers.
    unique_marker = np.unique(marker)
    unique_marker.sort()
    unique_marker = unique_marker[unique_marker != "none"]

    unique_select = np.unique(select)
    unique_select.sort()

    # Get all marker/select combos seen in the data
    combos_seen = set(list(zip(marker,select)))

    if param_names is None:

          # Always have a base growth intercept and slope
        param_names = ["base|b",
                       "base|m"]

        # Create a parameter intercept/slope pair for every marker,selector 
        # combo seen. 
        for m in unique_marker:
            for s in unique_select:
                if (m,s) in combos_seen:
                    param_names.append(f"{m}|{s}|b")
                    param_names.append(f"{m}|{s}|m")


    param_name_dict = dict([(p,i) for i, p in enumerate(param_names)])

    # Build the empty design matrix
    num_params = len(param_names) 
    num_samples = len(iptg)
    X = np.zeros((num_samples,num_params),dtype=float)

    # This indexer can be masked to select specific rows
    row_indexer = np.arange(num_samples,dtype=int)

    # If theta was not specified, calculate it from K and n
    if theta is None:

        if K is None or n is None:
            err = "K and n must be specified if theta is not\n"
            raise ValueError(err)
        
        K = float(K)
        n = float(n)
        theta = 1 - (iptg**n)/(K**n + iptg**n)

    # base growth intercept (@ log(iptg + log_iptg_offset) == 0)
    X[:,0] = 1

    # base growth slope. The independent variable is log_iptg. 
    X[:,1] = np.log(iptg + float(log_iptg_offset))

    # Now go through remaining combinations and add slope and intercept 
    # parameters. The independent variable is theta. 
    for m in unique_marker:
        for s in unique_select:
            if (m,s) in combos_seen:

                b_idx = param_name_dict[f"{m}|{s}|b"]
                m_idx = param_name_dict[f"{m}|{s}|m"]

                mask = np.logical_and(marker == m,select == s)
                X[row_indexer[mask],b_idx] = 1
                X[row_indexer[mask],m_idx] = theta[mask]
                
    return param_names, X