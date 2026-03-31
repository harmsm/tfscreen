==========
Simulation
==========

The `tfscreen.simulate` module provides tools for simulating high-throughput screens of transcription factor libraries.

Overview
--------
The simulation pipeline typically involves:
1.  **Defining a thermodynamic model**: Specifying binding sites, energy parameters, and how they map to transcriptional activity.
2.  **Predicting library activity**: Calculating the expected activity for all variants in the library under different conditions.
3.  **Converting activity to growth**: Mapping biochemical activity to bacterial growth rates using empirical or theoretical relationships.
4.  **Simulating selection experiments**: Modeling the change in variant frequencies over time during competition experiments.

Key Components
--------------
*   `selection_experiment`: Core logic for simulating a multi-round selection process.
*   `thermo_to_growth`: Functions for relating transcriptional activity to fitness.
*   `library_prediction`: Tools for applying a model to a full sequence library.
