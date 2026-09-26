---
title: Measurements that fix each tube's total population, especially under strong selection
status: idea
filed: 2026-09-26
area: experiment
revisit_when: >-
  Now for the few bench experiments the current dataset can still use
  (they feed roadmap steps 5-6); again when the next screen is designed
  (earlier timepoints, longer time base).
related:
  - planning/analysis-roadmap.md
  - docs/source/process-raw.rst
  - src/tfscreen/tfmodel/generative/observe/base_growth.py
---

**Context:** The analysis roadmap (`planning/analysis-roadmap.md`, "Reads and
totals") builds each tube's total population `N_s` into the fit as a smooth
curve observed through OD600. Reads measure only frequencies, so the total is
the sole source of the absolute scale, and with it the condition growth rates
`k_c` and, in the relative fit, `m_c`. Two gaps (user, 2026-09-26):

- **OD600 has no signal under the strongest selection.** Tubes that barely
  grow (particularly kan) stay below the reader's detection threshold
  (OD 0.096) at the timepoints we have. Censoring turns such a reading into
  an upper bound, not a measurement.
- **The next screen should sample earlier**, to resolve growth over a longer
  time base, and early tubes are below the threshold too.

What already constrains a blind tube:

- **The start.** The culture is grown to a presplit OD600, then diluted by
  the same factor into every tube, so every tube of a replicate starts from
  a known total (calibration plus pipetting error), shared by all of them.
- **Later, visible tubes of the same curve.** A smooth curve through a known
  start and later detectable points interpolates the early blind ones. This
  covers the future design if late points stay visible.
- **Other columns of the same condition.** `k_c` and `m_c` are shared by all
  IPTG columns of a condition and each genotype's X is a smooth Hill curve
  across columns, so a total that is visible in some columns constrains the
  others through the reads. This fails when every column of a condition is
  blind at a timepoint (then `k_c` trades freely with the curves), and it
  leans on the Hill shape exactly where the gauge needs it most (wt at 0 and
  1 mM).

**Idea:** add measurements of the absolute scale that do not depend on
OD600, chosen by what each one pins.

*For the current dataset (a few targeted experiments):*

1. **wt monoculture growth rates in each selection condition at the gauge
   concentrations** (0 and 1 mM IPTG), in the screen's tube format and
   volume, started at the screen's split density (known from presplit OD
   and dilution; aminoglycoside killing depends on inoculum density, so
   matching it matters), counted by plating over the screen's time window.
   Three replicates. In the relative fit this measures `k_c` (wt at 1 mM)
   and `m_c` (wt at 0 minus wt at 1 mM) directly: the gauge stops depending
   on the blind tubes. Add one or two controls: a non-binding repressor
   (IPTG-independent X, the internal standard of roadmap C4) and, if one
   exists, a super-repressor. Needs a new observer (roadmap step 6): the
   clean growth rate `k_c + dk_g + m_c X_g(c)` of a named genotype in a
   named condition and concentration. Today's `base_growth` observes only
   `k_ref + dk_geno` in one reference condition. Caveat: a monoculture
   lacks the library's other cells (density, any cross-protection), which
   the matched starting density only partly addresses.
   **Extended after study 0a (2026-09-26):** run the five genotypes with
   in vitro binding data (wt, M42I, H74A, K84L, M42I/H74A) across all eight
   IPTG concentrations in kan and 4CP. That measures the growth-binding
   map directly, with no library and no totals problem. 0a found the
   library data inconsistent with one linear map at the in vitro
   concentration scale (kan responds at roughly 30-100x lower IPTG; 4CP
   does not track binding), but could not rule out errors in the
   per-column totals.
2. **OD600 against plate counts under selection**, a few screen-like tubes
   in the strongest conditions at densities the reader can see. Tests
   whether the unstressed calibration transfers to stressed cells: dead
   cells and filaments scatter light without forming colonies (roadmap
   D11). Roughly ten to fifteen plates.
3. **qPCR of total plasmid in the extracted DNA, if any remains.** Relative
   plasmid amount per sample, calibrated against the samples whose OD600 was
   visible, gives relative totals for the blind ones. Limited by extraction
   yield and copy number varying between samples, so it is a weak
   constraint, but it reaches where OD600 cannot. Only if the DNA exists.

*For the next screen (built into the design):*

4. **A cell spike-in counting standard.** At each pull, before spin-down,
   pipette a fixed volume of a quantified frozen stock of cells carrying a
   plasmid with the same amplicon and a unique sequence absent from the
   library (for example a synonymous-codon wt, the same trick planned for
   telling spikes from bulk). Then `N_s = N_std * reads_library /
   reads_std`, at any density sequencing can see, with no plates and one
   pipetting step per tube. It is co-extracted and co-amplified with the
   library, so extraction and PCR differences cancel; the stock is
   quantified once, carefully, by plating. Choose `N_std` so the standard
   takes about 1-5% of reads. It also cross-checks OD600 wherever both
   exist. In the model it is one more "genotype" with a known, non-growing
   abundance per tube, and it fixes `P_s` directly.
5. **Flow cytometry with counting beads** (and a live/dead stain), as an
   alternative or check: absolute counts well below OD600's floor, and it
   separates live from dead cells. Costs instrument time per tube.
6. Keep OD600 for the visible tubes and the presplit; it is free and the
   calibration exists.

**Why not now:** Filed while revising the roadmap; the user will choose which
bench experiments are feasible for the current analysis (a few, not many).

**What it would take:**

- Bench time for 1-3 (1 is the most valuable for the relative fit).
- Model: the monoculture growth observer (roadmap step 6); a spike-in
  standard is a small addition to the count likelihood (roadmap step 7).
- Simulator: monoculture rates and a spike-in standard as optional outputs,
  to size how much each helps before running it (roadmap step 4).
- Open question for 1: which conditions are blind at which timepoints in the
  current data (roadmap study 0c answers it), so the experiment covers
  exactly those.
