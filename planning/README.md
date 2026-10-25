# planning/

Three kinds of things live here, kept together so they can link to each
other:

- **Ideas** (`status: idea`): tasks that matter but are deliberately out of
  scope for current work. Filing an idea here means "we decided not to do
  this now," not "we will do this."
- **Plans** (`status: active`): the few agreed multi-step plans being worked
  now. Keeping them next to the ideas they spin off means an idea's `related`
  link and a plan's step list can point at each other.
- **Studies** (`studies/`): the numerical or simulation work that answers a
  question a plan or idea raised. See "Studies" below.

Ideas and plans are one Markdown file each at the top level; the status in the
header tells them apart. (This directory was called `future/` until
2026-09-23.)

## Why a directory and not issues

GitHub issues are the usual home for a backlog, but they live outside the
repository: they aren't versioned with the code they describe, can't be read
offline, and can't be read by tools working in a checkout. These files are
plain Markdown with a small YAML header, so both people and scripts can list
and filter them.

## File format

Name files with short kebab-case slugs (`estimate-dk-alpha-by-varying-lambda.md`).
Start each file with this header:

```yaml
---
title: One-line name of the idea
status: idea        # idea | active | promoted | dropped | done
filed: 2026-09-13   # date filed (YYYY-MM-DD)
area: tfmodel       # tfmodel | simulate | analysis | process_raw | experiment | docs
revisit_when: >-
  The condition under which this becomes worth doing.
related:            # repo paths or other planning/ files; may be empty
  - src/tfscreen/simulate/selection_experiment.py
---
```

Then the body, with these sections (an `active` plan may use its own
structure, such as assumptions, decisions and ordered steps):

- **Context:** what we were doing when the idea came up.
- **Idea:** what to do.
- **Why not now:** the reason it was deferred.
- **What it would take:** rough scope, dependencies, open questions.

## Lifecycle

- `idea`: filed and waiting.
- `active`: an agreed plan being worked now. Keep its step list current.
- `done`: an active plan that is finished. Add a line saying where the result
  lives.
- `promoted`: work started. Add a line saying where it went (branch, changelog
  entry, plan). Keep the file as a record.
- `dropped`: decided against. Add a line saying why. Keep the file so the idea
  isn't re-proposed without that context.

To list open ideas:

```bash
grep -l "^status: idea" planning/*.md
```

## Studies

A study answers one question that a plan or idea raised, and its result feeds
a decision. Each study gets its own directory, `studies/<slug>/`, holding at
least:

- `README.md` with:
  - **Question** and the **decision it fed**, linking the plan step.
  - **How to run** it, from a scratch directory.
  - **Inputs**: repository paths only, with fixed seeds.
  - **Commit** the recorded results were produced on.
  - **Results**: the numbers the decision rests on.
- the script(s) that produce those results.

Rules:

- The plan records the decision and a short summary; the study README keeps
  the full numbers. Link both ways.
- Generated outputs (CSVs, figures) are not committed unless small and cited;
  the README's results are the record.
- Once a plan cites a study, freeze it. If the question changes, write a new
  study rather than rewriting the old one, so the citation stays valid.
- `dev/` stays untracked scratch space. Anything a plan cites moves here.
