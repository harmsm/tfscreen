---
title: Combination-specific effects of co-expressed variants on protein level
status: idea
filed: 2026-09-13
area: tfmodel
revisit_when: >-
  If double-transformation or native mass spec experiments show effects that
  depend on the specific pair of variants rather than on each variant alone.
related:
  - future/congression-physics-plan.md
---

**Context:** The congression plan assumes total repressor expression per cell
does not depend on how many plasmids a cell carries, and that each variant
contributes independently (a share-weighted theta and a soft-min dk).

**Idea:** Specific combinations could break this. For example, a very
unstable heterodimer might trigger a strong unfolded-protein response that
lowers the functional protein concentration or adds a growth cost beyond
either variant alone. That is a genotype-by-genotype interaction that feeds
back onto protein concentration, and so onto both theta and dk.

**Why not now:** Not resolvable with a single-lambda screen, where any given
pair of genotypes almost never shares a cell, and there is no evidence yet
that it matters.

**What it would take:** Bench data first (pairs co-expressed, protein levels
and growth measured). A model would need pair-specific terms, which is only
plausible for a small, targeted set of pairs.
