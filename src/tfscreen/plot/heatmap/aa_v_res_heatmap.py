import tfscreen
from tfscreen.util.dataframe import check_columns

import numpy as np

def aa_v_res_heatmap(single_df,
                     value_column,
                     heatmap_kwargs=None):

    # Expand heatmap_kwargs if needed
    if heatmap_kwargs is None:
        heatmap_kwargs = {}

    # Set x-axis and y-axis types if the user didn't specify
    heatmap_kwargs = heatmap_kwargs | {"x_axis_type":"site",
                                       "y_axis_type":"aa"}
        
    # Make sure the column has all needed columns
    check_columns(single_df,["genotype",
                             "resid",
                             "mut_aa",
                             value_column])
    
    # Make sure the genotypes are all unique
    if np.any(single_df["genotype"].duplicated()):
        raise ValueError (
            "genotypes must be unique to plot an aa vs. genotype heat map."
        )

    # Put amino acids in columns and residues in rows.
    for_hm = single_df.pivot_table(index="resid",
                                    columns="mut_aa",
                                    values=value_column)
    # Reverse-sort on columns so amino acids go top to bottom sorted
    for_hm = for_hm[for_hm.columns[::-1]]
    
    # Make sure residues are continuous
    for_hm = for_hm.reindex(np.arange(np.min(for_hm.index),
                                      np.max(for_hm.index)+1))

    # Generate heatmap
    fig, ax = tfscreen.plot.heatmap(for_hm,**heatmap_kwargs)

    return fig, ax