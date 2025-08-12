import numpy as np

def process_counts(sequence_counts,
                   total_counts,
                   total_cfu_ml,
                   pseudocount=1):
    """
    Convert sequence counts to frequencies and variances. 

    Parameters
    ----------
    sequence_counts : numpy.ndarray
        Array of sequence counts for a specific genotype/condition.
    total_counts : numpy.ndarray
        Array of total sequence counts for each sample.
    total_cfu_ml : numpy.ndarray
        Array of total CFU/mL measurements for each sample.
    pseudocount : int or float, optional
        Pseudocount added to sequence counts to avoid division by zero. Default is 1.

    Returns
    -------
    out : dict
        Dictionary containing the following keys:
        - "adj_seq_counts" : numpy.ndarray
            Adjusted sequence counts (sequence_counts + pseudocount).
        - "adj_total_counts" : numpy.ndarray
            Adjusted total counts (total_counts + n*pseudocount).
        - "f" : numpy.ndarray
            Frequency of the genotype (adj_sequence_counts / adj_total_counts).
        - "f_var" : numpy.ndarray
            Variance of the frequency.
        - "cfu" : numpy.ndarray
            CFU/mL of the genotype (f * total_cfu_ml).
        - "cfu_var" : numpy.ndarray
            Variance of the CFU/mL.
        - "ln_f" : numpy.ndarray
            Natural logarithm of the frequency.
        - "ln_f_var" : numpy.ndarray
            Variance of the natural logarithm of the frequency.
        - "ln_cfu" : numpy.ndarray
            Natural logarithm of the CFU/mL.
        - "ln_cfu_var" : numpy.ndarray
            Variance of the natural logarithm of the CFU/mL.    
    """

    n = len(sequence_counts)

    adj_sequence_counts = sequence_counts + pseudocount
    adj_total_counts = total_counts + n*pseudocount
    
    f = (adj_sequence_counts)/(adj_total_counts)
    f_var = f*(1 - f)/(adj_total_counts)
    
    cfu = f*total_cfu_ml
    cfu_var = f_var*(total_cfu_ml**2)
    
    ln_f = np.log(f)
    ln_f_var = f_var/(f**2)
    
    ln_cfu = np.log(cfu)
    ln_cfu_var = cfu_var/(cfu**2)

    out = {"adj_seq_counts":adj_sequence_counts,
           "adj_total_counts":adj_total_counts,
           "f":f,
           "f_var":f_var,
           "cfu":cfu,
           "cfu_var":cfu_var,
           "ln_f":ln_f,
           "ln_f_var":ln_f_var,
           "ln_cfu":ln_cfu,
           "ln_cfu_var":ln_cfu_var}

    return out