import tfscreen
from tfscreen.util import check_columns

import numpy as np

def aa_v_titrant_heatmap(single_df,
                         titrant_column,
                         value_column,
                         heatmap_kwargs=None):

    # Expand heatmap_kwargs if needed
    if heatmap_kwargs is None:
        heatmap_kwargs = {}

    # Set x-axis and y-axis types if the user didn't specify
    heatmap_kwargs = heatmap_kwargs | {"x_axis_type":"titrant",
                                       "y_axis_type":"aa"}
        
    # Make sure the column has all needed columns
    check_columns(single_df,["resid",
                             "mut_aa",
                             titrant_column,
                             value_column])
    
    # Make sure correct columns are unique
    if np.any(single_df[["mut_aa",titrant_column]].duplicated()):
        raise ValueError (
            "residue/titrant must be unique to plot an aa vs. titrant heat map."
        )

    # Put amino acids in columns and residues in rows.
    for_hm = single_df.pivot_table(index=titrant_column,
                                   columns="mut_aa",
                                   values=value_column)
    # Reverse-sort on columns so amino acids go top to bottom sorted
    for_hm = for_hm[for_hm.columns[::-1]]
    
    # Generate heatmap
    fig, ax = tfscreen.plot.heatmap(for_hm,**heatmap_kwargs)

    return fig, ax