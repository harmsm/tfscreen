import tfscreen
from tfscreen.util import check_columns

import numpy as np

def epistasis_heatmap(df,r1,r2,
                      value_column,
                      heatmap_kwargs=None):

    # Expand heatmap_kwargs if needed
    if heatmap_kwargs is None:
        heatmap_kwargs = {}

    # Set x-axis and y-axis types if the user didn't specify
    heatmap_kwargs = heatmap_kwargs | {"x_axis_type":"aa",
                                       "y_axis_type":"aa"}

    # Make sure the column has all needed columns
    check_columns(df,["num_muts",
                      "resid_1","resid_2",
                      "mut_aa_1","mut_aa_2",
                      value_column])
    
    # Create a dataframe with wt, resid 1, resid 2, and resid 1/2
    resids = [r1,r2]
    zero = df["num_muts"] == 0
    single = ((df["resid_1"].isin(resids) | df["resid_2"].isin(resids)) & (df["num_muts"] == 1))
    double = (df["resid_1"].isin(resids) & df["resid_2"].isin(resids))
    sub_df = df[zero | single | double]

    # Make sure the genotypes are all unique
    if np.any(df[zero]["genotype"].duplicated()):
        raise ValueError (
            "condition_selector must be unique to plot an epistasis heat map."
        )
    
    # Extract epistasis
    ep_df = tfscreen.analysis.extract_epistasis(sub_df,
                                                condition_selector=None, #condition_selector,
                                                y_obs=value_column)
    ep_df = tfscreen.genetics.expand_genotype_columns(ep_df)

    # Pivot for heat map creation
    for_hm = ep_df.pivot_table(index="mut_aa_1",
                               columns="mut_aa_2",
                               values="ep_obs")
    for_hm = for_hm[for_hm.columns[::-1]]

    # Make heat map
    fig, ax = tfscreen.plot.heatmap(for_hm,**heatmap_kwargs)

    return fig, ax