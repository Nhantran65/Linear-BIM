# Canonical experiment recipes

These recipes are the reviewed entry points for future runs. They use
`src/linear_bim`, explicit paper configs, supplied data/cache paths, and fresh
`runs/<run-id>` outputs. They do not import historical packages and never
overwrite `artifacts/frozen`.

| Recipe | Runner | Config | Historical authority |
|---|---|---|---|
| S20 | `sleep_ulw_author_literal_bim/run_experiment.py` | `configs/paper/s20.yaml` | `sleep_ulw_author_literal_bim` |
| S78 | `sleep_ulw_s78_author/run.py` | `configs/paper/s78.yaml` | `sleep_ulw_s78_author` |
| S3 | `sleep_cvan_matched_benchmark/run_linear_bim.py` | `configs/paper/s3.yaml` | `sleep_ulw_isruc_s1/models.py` + `sleep_cvan_matched_benchmark` |

The historical runner names above are provenance mappings only while the
reorganized runners are under review. The current supported CLI stage is
explicit CPU/GPU BIM-head training over supplied embeddings and logits; full
backbone replay requires a later reviewed implementation.
