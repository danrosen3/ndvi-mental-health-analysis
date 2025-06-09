# NDVI & Mental Health Analysis

This repository contains the full analysis pipeline examining the relationship between early adulthood green space exposure (measured using NDVI — Normalized Difference Vegetation Index) and later-life risk of dementia and depression among older adults.

The analysis uses data from the **All of Us Research Program Controlled Tier Dataset (v8)** and integrates geospatial, clinical, and socioeconomic information. Results are summarized in figures and tables suitable for publication.

---

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt

2. Run the analysis:

```bash
python ndvi_dementia_analysis_pipeline.py

---

## Key Outputs

- **Figure 2** – NDVI dose–response spline
- **Figure 3** – ZIP3-level spatial NDVI map
- **Figure 4** – NDVI stratified by SES
- **Smooth_Difference_by_SES.pdf** – Risk difference by SES
- **Table 1** – NDVI cohort characteristics
- **Table 2** – Logistic regression results

---

## Spatial Mapping Shapefiles

Shapefiles for ZIP3 mapping are not included in this repository.

Download them here:  
[Google Drive – NDVI_Shapefiles_ZIP3](https://drive.google.com/drive/folders/19BiQDEKbYPkRJN8mHImNF3-x7sN7PvQ3?usp=drive_link)

Included files:
- `three_dig_zips.shp`
- `three_dig_zips.shx`
- `three_dig_zips.dbf`
- `three_dig_zips.prj`

---

## Data Access

This analysis used the **All of Us Controlled Tier Dataset v8**.  
You must be an authorized researcher to access these data:  
🔗 https://www.researchallofus.org

---

## License

This project is licensed under the MIT License.  
See `LICENSE` for details.

---

## Citation

Trabilsy M*, Rosenthal DA*, Barr P.
The Relationship Between Green Space Exposure in Early Adulthood and Neuropsychiatric Risk in Later Life: A Socioeconomically Stratified National Analysis Based on the "All of Us" Research Program.
SUNY Downstate Medical Center.
Authors marked with an asterisk contributed equally.
*(Manuscript under review)*

---

## Contact

danielr221196@gmail.com
