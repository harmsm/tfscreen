"""
``tfs-check-tubes``: flag selection tubes whose genotype composition
disagrees with their titrant label (a likely tube or sample-sheet swap).

See ``tfscreen.process_raw.tube_qc`` for the method.
"""

import pandas as pd

from tfscreen.process_raw.tube_qc import (
    PANEL_COLUMNS,
    ROW_COLUMNS,
    score_tubes,
    score_table,
)
from tfscreen.util.cli import generalized_main
from tfscreen.util.io import read_dataframe


def check_tubes(growth_df,
                out_prefix="tfs_tube_qc",
                min_counts=1,
                min_genotypes=10,
                flag_z=4.0):
    """
    Score every tube's apparent titrant from its genotype composition and
    flag tubes that are out of order within their row.

    Within each library x condition_pre x condition_sel x titrant_name
    panel, each tube's log-ratio composition has its row (replicate x t_pre
    x t_sel) effect removed by a robust fit (medians, with a per-row scale
    since the titrant's effect grows with time) and is projected on the
    axis from the lowest concentration's composition (score 0) to the
    highest's (score 1). Tube totals and read depth cancel, so OD600 and depth errors
    do not enter. A tube is flagged when its score is out of order with the
    rest of its row (leave-one-out isotonic fit) or far from its label's
    expected score, by more than ``flag_z`` noise SDs; rows of fewer than
    three tubes are not flagged. Rows missing some concentrations are
    handled without the shift of centering a row on itself.

    Writes ``{out_prefix}.csv``, one row per tube (``score``, ``expected``,
    ``order_z``, ``deviation_z``, ``flagged`` and more), and prints each
    panel's scores by row and concentration and every tube of the rows
    with a flagged tube. Once a row is down to two unflagged tubes neither
    is flagged, so read the whole row: a large ``order_z`` left there is
    also suspect.

    Parameters
    ----------
    growth_df : str
        Growth table from tfs-process-counts (any long-form table with
        ``genotype``, ``counts``, ``replicate`` and ``titrant_conc``).
    out_prefix : str
        Output prefix.
    min_counts : int
        Use a genotype only if it has at least this many reads in every tube
        of its panel.
    min_genotypes : int
        Fewest usable genotypes a panel may have.
    flag_z : float
        Flag a tube whose order violation or deviation from its expected
        score exceeds this many noise SDs.
    """
    df = read_dataframe(growth_df)
    scores = score_tubes(df, min_counts=min_counts,
                         min_genotypes=min_genotypes, flag_z=flag_z)

    with pd.option_context("display.width", 200,
                           "display.max_columns", 100):
        print("Apparent titrant score (0 = lowest concentration, "
              "1 = highest), by row and titrant_conc:")
        print(score_table(scores).round(2).to_string())
        n_flagged = int(scores["flagged"].sum())
        if n_flagged:
            # Show whole rows: a row's last two tubes are never flagged.
            keys = [c for c in PANEL_COLUMNS + ROW_COLUMNS
                    if c in scores.columns]
            bad = scores.groupby(keys)["flagged"].transform("any")
            rows = scores[bad]
            cols = [c for c in rows.columns
                    if c not in ("n_genotypes", "sigma", "row_scale",
                                 "deviation_sd")]
            stats = ["score", "expected", "deviation", "order_violation",
                     "order_z", "deviation_z"]
            print(f"\n{n_flagged} flagged tube(s); every tube of their rows:")
            print(rows[cols].round({c: 3 for c in stats})
                  .to_string(index=False))
        else:
            print("\nNo tubes flagged.")

    scores.to_csv(f"{out_prefix}.csv", index=False)


def main():
    generalized_main(check_tubes)


if __name__ == "__main__":
    main()
