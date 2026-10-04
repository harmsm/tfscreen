# Raw-data example

A small synthetic experiment in the formats `tfs-process-counts` reads. The
numbers are made up. They show the formats, not a realistic screen.

- `library_config.yaml`: the library description. The same file goes to
  `tfs-process-fastq` and to `tfs-configure-model --library_config`.
- `make_example_data.py`: writes the three inputs below from a fixed seed.
- `tube_table.csv`: one row per sequenced tube. Two libraries (kanR and
  pheS), each with a selection and a control condition, two IPTG
  concentrations and three time points, for 24 tubes.
- `od600.csv`: one OD600 reading per tube.
- `counts/`: one count file per tube, in the format `tfs-process-fastq`
  writes, including the `__unknown__` row.

The OD600 calibration is in `../od600/`. To build the growth table:

```bash
python make_example_data.py
tfs-process-counts tube_table.csv counts \
    --od600_file od600.csv \
    --od600_calibration_file ../od600/od600_calibration.yaml \
    --tube_volume_mL 5 \
    --out_prefix growth
```

`growth.csv` is ready for
`tfs-configure-model --growth_df growth.csv --library_config library_config.yaml`.

See `docs/source/process-raw.rst` for the formats and the steps.
