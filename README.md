# NDVI & Mental Health Analysis

This repository contains the full analysis pipeline examining the relationship between early adulthood green space exposure (NDVI) and later-life risk of **dementia** and **depression**.

The analysis uses the **All of Us Controlled Tier Dataset v8** and integrates geospatial, clinical, and behavioral data.

---

## Contents

- `ndvi_dementia_analysis_pipeline.py` — full analysis script (from raw data to figures)
- `results/` — final figures and tables:
  - `figure2_ndvi_dose_response_spline.pdf`
  - `figure3_ndvi_spatial_map_fixed.pdf`
  - `figure4_ndvi_effect_modification.pdf`
  - `table1_ndvi_cohort.html`
  - `table2_multivariable_logistic.csv`
- `data/` — input files used in the pipeline (e.g. ZIP3 NDVI)
- `ndvi_manuscript_outputs.zip` — full bundle of all outputs

---

## Summary of Key Findings

- Higher NDVI exposure (ages 18–25) was associated with reduced risk of dementia/depression.
- SES modifies the relationship — stronger protective effects observed in low-deprivation ZIPs.
- A spatial ZIP3 NDVI map was generated for all participants.

---

## Reproducibility

To run this pipeline:

```bash
pip install -r requirements.txt
python ndvi_dementia_analysis_pipeline.py
