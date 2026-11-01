"""
General thermodynamic model of a dimeric transcription factor.

``species.py`` holds the physics (species table and solver), ``model.py``
the parts the three prior variants share, and ``PK.py``, ``PddG.py`` and
``PnnC.py`` the variants registered as ``thermo.dimer.PK``,
``thermo.dimer.PddG`` and ``thermo.dimer.PnnC``.  Design and study:
``planning/dimer-thermo-model.md`` and
``planning/studies/thermo-identifiability/``.
"""
