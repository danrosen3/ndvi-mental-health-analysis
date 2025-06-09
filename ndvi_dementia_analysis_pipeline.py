#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.environ["GOOGLE_CLOUD_PROJECT"] = "terra-vpc-sc-a40a3f9a"
os.environ["WORKSPACE_CDR"] = "fc-aou-cdr-prod-ct.C2024Q3R5"


# In[2]:


get_ipython().run_line_magic('pip', 'install --force-reinstall --no-cache-dir      "numpy==1.24.4"      "pandas==1.5.3"      "protobuf==3.20.2"      "scipy<1.12"      "matplotlib<=3.7.3"      "pyarrow<10"      "pandas-gbq==0.19.2"      "google-cloud-bigquery==3.4.2"      "google-cloud-bigquery-storage==2.16.2"      "pygam==0.8.0"')


# In[1]:


import sys
import shutil
import os

for path in sys.path:
    maybe = os.path.join(path, "google/cloud/__init__.py")
    if os.path.exists(maybe):
        print("Removing shadowed Google path:", path)
        shutil.rmtree(os.path.join(path, "google"), ignore_errors=True)


# In[2]:


get_ipython().run_line_magic('pip', 'install --force-reinstall google-cloud-bigquery==3.4.2')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib
import pyarrow
import google.cloud.bigquery
import pandas_gbq
import pygam
import scipy

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("pyarrow:", pyarrow.__version__)
print("bigquery:", google.cloud.bigquery.__version__)
print("pandas-gbq:", pandas_gbq.__version__)
print("pygam:", pygam.__version__)
print("scipy:", scipy.__version__)


# In[4]:


# Dementia
dementia_ids = [
    37312035, 44784643, 378419, 35608576, 37117145, 4092747, 4182210, 4228133, 378726, 374888, 40483103,
    441002, 43530664, 43530666, 380701, 376095, 44782432, 4043378, 377788, 37109222, 4139421, 44782763,
    762497, 4046090, 43021816, 762704, 379778, 444091, 443790, 443864, 377254, 44784521, 378125, 381832,
    44782771, 377527, 4218017, 4277444, 4220313, 4278830, 4043379, 4103534, 4009647, 373179, 4048875,
    4196433, 376946, 380986, 379784, 4101137, 765653, 4047747, 376085, 375791, 443605, 37018688, 37109056
]

# Depression
depression_ids = [
    438727, 440698, 4094358, 440383, 44782943, 3656234, 433440, 4282096, 4152280, 4174987, 4338031,
    607543, 433751, 4282316, 35615154, 35615155, 432285, 438998, 432883, 434911, 438406, 4191716
]

# Exclusion
exclude_ids = [
    439703, 436071, 433734, 437243, 435236, 4100366, 439275, 439274, 434901, 432865, 4046108, 4133495,
    36715010, 36716783, 439776, 439780, 4231948, 439254, 440078, 435226, 439251, 439253, 437528, 439255,
    441834, 433992, 439256, 35624745, 436665, 4220618, 436072, 4310821, 35624743, 35624744, 432876,
    4071442, 35624748, 4150985, 35624747, 432866, 4148842, 4327669, 4307956, 4172156, 4037669, 4185096,
    4144519, 4148934, 37109940, 37117177, 4299505, 4168389, 433996, 434332, 4231949, 4197669, 4001733,
    441538, 440368, 436385, 435235, 436944, 436673, 435218, 441835, 435782, 4105330, 761978, 40483103,
    441836, 435225, 4177651, 4201739, 441828, 4102337, 45765723, 374341, 4177039, 4254211, 40277917,
    4041136, 433742, 432300, 4287544, 436086, 442600, 4166701, 4028027, 432290, 4324945, 437250, 432612,
    4215917, 440079, 439250, 439245, 439249, 439248, 439246, 443906, 4009648, 437529, 433743, 4194222,
    440067, 4262111, 4280361, 436682, 4307804, 439785, 374919, 4248716, 36713737, 433450, 435217, 381270,
    4126631, 4204820, 4140090, 4171569, 4064308, 4178929, 438733, 37311816, 37110514, 35622934, 37312578,
    4145049, 439702, 439004, 440686, 4286201, 4244078, 4224940, 4152971, 432597, 434321, 435783, 435219,
    4085662, 374013, 4137855, 4155798, 4195158, 4131027, 4200385, 4154283, 4220617, 4030856, 4161200,
    4217940, 4045263, 4262272, 42872413, 436386, 442570, 432898, 43020451, 4102603, 4141603, 443797,
    42872412, 439001, 372599, 436067, 433990, 438724, 436384, 440077, 432299, 433442, 444396, 433443,
    440373, 432598, 4310121, 4140881, 4008566, 4046093, 37395785
]

all_condition_ids = list(set(dementia_ids + depression_ids + exclude_ids))


# In[5]:


from pandas_gbq import read_gbq
cdr = os.environ["WORKSPACE_CDR"]

condition_sql = f"""
SELECT person_id, condition_concept_id, condition_start_datetime
FROM `{cdr}.condition_occurrence`
WHERE condition_concept_id IN ({','.join(map(str, all_condition_ids))})
"""

condition_df = read_gbq(condition_sql, use_bqstorage_api=True)
print("Conditions loaded:", condition_df.shape)


# In[6]:


person_sql = f"""
SELECT
  person_id,
  concept_gender.concept_name AS gender,
  birth_datetime AS date_of_birth,
  concept_race.concept_name AS race,
  concept_eth.concept_name AS ethnicity
FROM `{cdr}.person`
LEFT JOIN `{cdr}.concept` concept_gender ON gender_concept_id = concept_gender.concept_id
LEFT JOIN `{cdr}.concept` concept_race ON race_concept_id = concept_race.concept_id
LEFT JOIN `{cdr}.concept` concept_eth ON ethnicity_concept_id = concept_eth.concept_id
"""

person_df = read_gbq(person_sql, use_bqstorage_api=True)
print("Demographics loaded:", person_df.shape)


# In[7]:


zip_sql = f"""
SELECT
  obs.person_id,
  zip.zip3_as_string AS zip3,
  zip.median_income,
  zip.fraction_poverty AS poverty,
  zip.deprivation_index
FROM `{cdr}.zip3_ses_map` zip
JOIN `{cdr}.observation` obs
  ON CAST(SUBSTR(obs.value_as_string, 0, STRPOS(obs.value_as_string, '*') - 1) AS INT64) = zip.zip3
WHERE obs.observation_source_concept_id = 1585250
  AND obs.value_as_string NOT LIKE 'Res%'
"""

zip_df = read_gbq(zip_sql, use_bqstorage_api=True)
zip_df["zip3"] = zip_df["zip3"].astype(str)
print("ZIP3 SES data loaded:", zip_df.shape)


# In[8]:


import pandas as pd
from datetime import date

# Convert to datetime
person_df["birth_year"] = pd.to_datetime(person_df["date_of_birth"]).dt.year
condition_df["condition_year"] = pd.to_datetime(condition_df["condition_start_datetime"]).dt.year

# Merge to compute age at diagnosis
merged = condition_df.merge(person_df[["person_id", "birth_year"]], on="person_id", how="left")
merged["age_at_dx"] = merged["condition_year"] - merged["birth_year"]

# Keep only those diagnosed at age 60+
merged = merged[merged["age_at_dx"] >= 60]


# In[9]:


# Ensure valid integers
merged = merged[merged["condition_concept_id"].notna()].copy()
merged["condition_concept_id"] = merged["condition_concept_id"].astype(int)

merged["is_dementia"] = merged["condition_concept_id"].isin(dementia_ids).astype(int)
merged["is_depression"] = merged["condition_concept_id"].isin(depression_ids).astype(int)
merged["is_excluded"] = merged["condition_concept_id"].isin(exclude_ids).astype(int)


# In[10]:


case_df = (
    merged.groupby("person_id")[["is_dementia", "is_depression", "is_excluded"]]
    .max()
    .reset_index()
)

# Keep only dementia or depression, not excluded
case_df = case_df[
    (case_df["is_excluded"] == 0) &
    ((case_df["is_dementia"] == 1) | (case_df["is_depression"] == 1))
]


# In[11]:


person_df["age"] = date.today().year - pd.to_datetime(person_df["date_of_birth"]).dt.year

excluded_ids = set(merged[merged["is_excluded"] == 1]["person_id"].unique())
case_ids = set(case_df["person_id"].unique())

control_df = person_df[
    (person_df["age"] >= 60) &
    ~person_df["person_id"].isin(case_ids) &
    ~person_df["person_id"].isin(excluded_ids)
].copy()

# Label controls
control_df["is_dementia"] = 0
control_df["is_depression"] = 0


# In[12]:


case_df = case_df.merge(person_df, on="person_id", how="left")
case_df = case_df.merge(zip_df, on="person_id", how="left")
control_df = control_df.merge(zip_df, on="person_id", how="left")


# In[13]:


analysis_df = pd.concat([case_df, control_df], ignore_index=True)
print("Combined cohort:", analysis_df.shape)


# In[14]:


from datetime import date

# Estimate birth year and NDVI age window
analysis_df["birth_year"] = date.today().year - analysis_df["age"]
analysis_df["year_18"] = analysis_df["birth_year"] + 18
analysis_df["year_25"] = analysis_df["birth_year"] + 25


# In[15]:


# Extract 3-digit ZIP from masked ZIPs like "024**"
analysis_df["zip3_clean"] = (
    analysis_df["zip3"]
    .astype(str)
    .str.extract(r"(\d{1,3})")[0]
    .str.zfill(3)
)


# In[16]:


ndvi_df = pd.read_csv("ndvi_by_zip3_year_filtered.csv")
print("NDVI file loaded:", ndvi_df.shape)


# In[17]:


ndvi_df["zip3"] = ndvi_df["zip3"].astype(str).str.zfill(3)


# In[18]:


expanded_rows = []

for _, row in analysis_df.iterrows():
    for year in range(int(row["year_18"]), int(row["year_25"]) + 1):
        expanded_rows.append({
            "person_id": row["person_id"],
            "zip3": row["zip3_clean"],
            "year": year
        })

exposure_df = pd.DataFrame(expanded_rows)


# In[19]:


exposure_df = exposure_df.merge(ndvi_df, on=["zip3", "year"], how="left")
print("NDVI values matched:", exposure_df["ndvi_value"].notna().sum())


# In[20]:


avg_ndvi = (
    exposure_df.groupby("person_id")["ndvi_value"]
    .mean()
    .reset_index()
    .rename(columns={"ndvi_value": "avg_ndvi_18_25"})
)


# In[21]:


analysis_df["person_id"] = analysis_df["person_id"].astype(str)
avg_ndvi["person_id"] = avg_ndvi["person_id"].astype(str)

analysis_df = analysis_df.merge(avg_ndvi, on="person_id", how="left")
print("NDVI exposure merged:", analysis_df["avg_ndvi_18_25"].notna().sum(), "non-null")


# In[22]:


analysis_df = analysis_df.dropna(subset=["avg_ndvi_18_25"]).copy()


# In[23]:


analysis_df["NDVI Quartile"] = pd.qcut(
    analysis_df["avg_ndvi_18_25"],
    4,
    labels=["Q1", "Q2", "Q3", "Q4"]
)


# In[24]:


summary = (
    analysis_df
    .groupby("NDVI Quartile", observed=False)[
        ["age", "median_income", "poverty", "deprivation_index"]
    ]
    .agg(["mean", "std", "count"])
    .round(2)
)


# In[25]:


summary.columns = [
    "Age (Mean)", "Age (SD)", "Age (Count)",
    "Income (Mean)", "Income (SD)", "Income (Count)",
    "Poverty % (Mean)", "Poverty % (SD)", "Poverty % (Count)",
    "Deprivation Index (Mean)", "Deprivation Index (SD)", "Deprivation Index (Count)"
]
summary = summary.reset_index()


# In[26]:


summary.to_html("table1_ndvi_cohort.html", index=False)
print("Saved HTML: table1_ndvi_cohort.html")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 6))
ax.axis("off")

tbl = ax.table(
    cellText=summary.values.astype(str).tolist(),
    colLabels=summary.columns.tolist(),
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(6)
tbl.scale(1.0, 1.5)

for cell in tbl.get_celld().values():
    cell.set_linewidth(0.2)
    cell.set_edgecolor("gray")

plt.tight_layout()
plt.savefig("table1_ndvi_cohort.pdf", dpi=300)
print("Saved PDF: table1_ndvi_cohort.pdf")


# In[27]:


analysis_df["has_disease"] = (
    (analysis_df["is_dementia"] == 1) | (analysis_df["is_depression"] == 1)
).astype(int)


# In[28]:


df_model = analysis_df.dropna(subset=["avg_ndvi_18_25", "has_disease"])


# In[29]:


import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

# Design matrices using cubic spline
y, X = patsy.dmatrices(
    "has_disease ~ cr(avg_ndvi_18_25, df=3)",
    data=df_model,
    return_type="dataframe"
)

# Fit logistic regression model
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()


# In[30]:


ndvi_vals = np.linspace(df_model["avg_ndvi_18_25"].min(), df_model["avg_ndvi_18_25"].max(), 100)
X_pred = patsy.dmatrix("cr(avg_ndvi_18_25, df=3)", data={"avg_ndvi_18_25": ndvi_vals}, return_type="dataframe")
pred = model.get_prediction(X_pred).summary_frame()

# Compute marginal slope
slope = np.gradient(pred["mean"], ndvi_vals)


# In[31]:


figure2_df = pd.DataFrame({
    "NDVI": ndvi_vals.round(3),
    "Predicted_Risk": pred["mean"].round(4),
    "CI_Lower": pred["mean_ci_lower"].round(4),
    "CI_Upper": pred["mean_ci_upper"].round(4),
    "Slope": slope.round(4)
})


# In[32]:


figure2_df.to_csv("figure2_ndvi_dose_response_data.csv", index=False)
figure2_df.to_html("figure2_ndvi_dose_response_data.html", index=False)
print("Saved: CSV + HTML for Figure 2")


# In[33]:


import matplotlib.pyplot as plt

plt.figure(figsize=(3.5, 2.75))  # Nature journal size
plt.plot(figure2_df["NDVI"], figure2_df["Predicted_Risk"], color="black", label="Spline")
plt.fill_between(
    figure2_df["NDVI"],
    figure2_df["CI_Lower"],
    figure2_df["CI_Upper"],
    color="gray",
    alpha=0.3
)
plt.xlabel("NDVI (Age 18–25)", fontsize=7)
plt.ylabel("Predicted Risk", fontsize=7)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.grid(False)
plt.tight_layout()
plt.savefig("figure2_ndvi_dose_response_spline.pdf", dpi=600)
plt.show()

print("Saved: figure2_ndvi_dose_response_spline.pdf")


# In[34]:


# Automatically detect common SES columns
ses_candidates = [col for col in analysis_df.columns if "deprivation" in col.lower() or "poverty" in col.lower()]
assert ses_candidates, "❌ No SES column found (expected: 'deprivation_index' or similar)."
ses_col = ses_candidates[0]
print("Using SES column:", ses_col)


# In[35]:


df_mod = analysis_df.dropna(subset=["avg_ndvi_18_25", "has_disease", ses_col]).copy()

df_mod["SES Group"] = np.where(
    df_mod[ses_col] > df_mod[ses_col].median(),
    "High Deprivation",
    "Low Deprivation"
)


# In[36]:


results = []

for group, group_df in df_mod.groupby("SES Group"):
    y, X = patsy.dmatrices("has_disease ~ cr(avg_ndvi_18_25, df=4)", data=group_df, return_type='dataframe')
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    ndvi_vals = np.linspace(group_df["avg_ndvi_18_25"].min(), group_df["avg_ndvi_18_25"].max(), 100)
    X_pred = patsy.dmatrix("cr(avg_ndvi_18_25, df=4)", data={"avg_ndvi_18_25": ndvi_vals}, return_type="dataframe")
    pred = model.get_prediction(X_pred).summary_frame()
    slope = np.gradient(pred["mean"], ndvi_vals)

    df_result = pd.DataFrame({
        "NDVI": ndvi_vals.round(3),
        "Predicted Risk": pred["mean"].round(4),
        "95% CI Lower": pred["mean_ci_lower"].round(4),
        "95% CI Upper": pred["mean_ci_upper"].round(4),
        "Slope (dRisk/dNDVI)": slope.round(4),
        "SES Group": group
    })

    results.append(df_result)

df_final = pd.concat(results, ignore_index=True)


# In[37]:


df_final.to_csv("figure4_ndvi_effect_modification_data.csv", index=False)
df_final.to_html("figure4_ndvi_effect_modification_data.html", index=False)
print("Saved: figure4_ndvi_effect_modification_data.csv + .html")


# In[38]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))

line_styles = {
    "Low Deprivation": "solid",
    "High Deprivation": "dashed"
}

for group in df_final["SES Group"].unique():
    g = df_final[df_final["SES Group"] == group]
    ax.plot(g["NDVI"], g["Predicted Risk"], linestyle=line_styles.get(group, "solid"), color="black", label=group)
    ax.fill_between(g["NDVI"], g["95% CI Lower"], g["95% CI Upper"], color="gray", alpha=0.2)

ax.set_title("NDVI–Disease Risk by SES Group")
ax.set_xlabel("NDVI (Age 18–25)")
ax.set_ylabel("Predicted Probability")

# 👇 Legend moved to upper-left
ax.legend(title="SES Group", loc="upper left")

plt.tight_layout()
plt.savefig("figure4_ndvi_effect_modification.pdf", dpi=300)
plt.show()


# In[39]:


ndvi_min = df_mod["avg_ndvi_18_25"].min()
ndvi_max = df_mod["avg_ndvi_18_25"].max()
ndvi_vals = np.linspace(ndvi_min, ndvi_max, 100)


# In[40]:


results = []

for group, group_df in df_mod.groupby("SES Group"):
    y, X = patsy.dmatrices("has_disease ~ cr(avg_ndvi_18_25, df=4)", data=group_df, return_type='dataframe')
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    df_pred = pd.DataFrame({"avg_ndvi_18_25": ndvi_vals})
    X_pred = patsy.dmatrix("cr(avg_ndvi_18_25, df=4)", data=df_pred, return_type="dataframe")
    pred = model.get_prediction(X_pred).summary_frame()
    slope = np.gradient(pred["mean"], ndvi_vals)

    results.append(pd.DataFrame({
        "NDVI": ndvi_vals,
        "Predicted Risk": pred["mean"],
        "CI Lower": pred["mean_ci_lower"],
        "CI Upper": pred["mean_ci_upper"],
        "Slope": slope,
        "SES Group": group
    }))


# In[41]:


df_aligned = pd.concat(results, ignore_index=True)

high_df = df_aligned[df_aligned["SES Group"] == "High Deprivation"]
low_df = df_aligned[df_aligned["SES Group"] == "Low Deprivation"]

# Now values will be aligned
diff = high_df["Predicted Risk"].values - low_df["Predicted Risk"].values

risk_diff_df = pd.DataFrame({
    "NDVI": ndvi_vals,
    "Risk_Difference_HighMinusLow": diff
})


# In[42]:


risk_diff_df.to_csv("Smooth_Difference_by_SES.csv", index=False)

plt.figure(figsize=(6, 4))
plt.plot(risk_diff_df["NDVI"], risk_diff_df["Risk_Difference_HighMinusLow"], color="black")
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("NDVI (Age 18–25)")
plt.ylabel("Risk Difference (High - Low)")
plt.title("NDVI–Risk Difference by SES Group")
plt.tight_layout()
plt.savefig("Smooth_Difference_by_SES.pdf", dpi=300)
plt.show()


# In[43]:


analysis_df["has_disease"] = (
    (analysis_df["is_dementia"] == 1) | (analysis_df["is_depression"] == 1)
).astype(int)


# In[44]:


analysis_df["gender_collapsed"] = analysis_df["gender"].where(
    analysis_df["gender"].isin(["Male", "Female"]), "Other"
)

analysis_df["race_collapsed"] = analysis_df["race"].where(
    analysis_df["race"].isin(["White", "Black or African American"]), "Other"
)


# In[45]:


for col in ["gender_collapsed", "race_collapsed"]:
    analysis_df[col] = analysis_df[col].astype("category")
    if "Missing" not in analysis_df[col].cat.categories:
        analysis_df[col] = analysis_df[col].cat.add_categories(["Missing"])
    analysis_df[col] = analysis_df[col].fillna("Missing")


# In[46]:


formula = (
    "has_disease ~ C(gender_collapsed, Treatment(reference='Male')) + "
    "C(race_collapsed, Treatment(reference='White')) + "
    "avg_ndvi_18_25 + age + deprivation_index"
)


# In[47]:


import patsy
import statsmodels.api as sm
import numpy as np

df_reg = analysis_df.dropna(subset=["avg_ndvi_18_25", "age", "has_disease", "deprivation_index"])
y, X = patsy.dmatrices(formula, data=df_reg, return_type="dataframe")
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()


# In[48]:


summary_df = model.summary2().tables[1].copy()
summary_df["Odds Ratio"] = np.exp(summary_df["Coef."])
summary_df["95% CI Lower"] = np.exp(summary_df["Coef."] - 1.96 * summary_df["Std.Err."])
summary_df["95% CI Upper"] = np.exp(summary_df["Coef."] + 1.96 * summary_df["Std.Err."])


# In[49]:


summary_df = summary_df.rename(columns={
    "Coef.": "Estimate (log-odds)",
    "Std.Err.": "Standard Error",
    "P>|z|": "P-value"
})[[
    "Estimate (log-odds)", "Standard Error", "P-value",
    "Odds Ratio", "95% CI Lower", "95% CI Upper"
]]

summary_df.index = summary_df.index.map(str)
summary_df.to_csv("table2_multivariable_logistic.csv")
summary_df.to_html("table2_multivariable_logistic.html")
print("Saved: Table 2 regression results (CSV + HTML)")


# In[50]:


summary_df.head(10)


# In[51]:


import geopandas as gpd

# Load and reproject shapefile
zip3_gdf = gpd.read_file("three_dig_zips.shp").to_crs("EPSG:4326")
zip3_gdf = zip3_gdf.rename(columns={"3dig_zip": "ZIP3"})
zip3_gdf["ZIP3"] = zip3_gdf["ZIP3"].astype(str).str.zfill(3)
print("Shapefile loaded:", zip3_gdf.shape)


# In[52]:


zip_ndvi = (
    analysis_df.dropna(subset=["avg_ndvi_18_25", "zip3_clean"])
    .groupby("zip3_clean")["avg_ndvi_18_25"]
    .mean()
    .reset_index()
    .rename(columns={"zip3_clean": "ZIP3", "avg_ndvi_18_25": "NDVI"})
)


# In[53]:


map_df = zip3_gdf.merge(zip_ndvi, on="ZIP3", how="left")


# In[54]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

map_df.plot(
    column="NDVI",
    cmap="Greys",  # Greyscale map for publication
    linewidth=0.2,
    edgecolor="black",
    legend=True,
    ax=ax,
    missing_kwds={"color": "lightgrey", "label": "Not in cohort"}
)

ax.set_title("Average NDVI Exposure (Age 18–25) by ZIP3", fontsize=9)
ax.axis("off")
ax.set_xlim([-130, -65])
ax.set_ylim([23, 50])
plt.tight_layout()
plt.savefig("figure3_ndvi_spatial_map_fixed.pdf", dpi=300)
plt.show()

print("Map saved: figure3_ndvi_spatial_map_fixed.pdf")


# In[55]:


# Recreate ZIP3 average NDVI table
zip_ndvi = (
    analysis_df.dropna(subset=["avg_ndvi_18_25", "zip3_clean"])
    .groupby("zip3_clean")["avg_ndvi_18_25"]
    .mean()
    .reset_index()
    .rename(columns={"zip3_clean": "ZIP3", "avg_ndvi_18_25": "NDVI"})
)

zip_ndvi["ZIP3"] = zip_ndvi["ZIP3"].astype(str).str.zfill(3)
zip_ndvi["NDVI"] = zip_ndvi["NDVI"].round(4)
zip_ndvi.to_csv("figure3_ndvi_by_zip3.csv", index=False)

print("Recreated: figure3_ndvi_by_zip3.csv")


# In[56]:


import zipfile
import os

output_files = [
    "table1_ndvi_cohort.html",
    "table2_multivariable_logistic.csv",
    "figure2_ndvi_dose_response_spline.pdf",
    "figure3_ndvi_spatial_map_fixed.pdf",
    "figure4_ndvi_effect_modification.pdf",
    "Smooth_Difference_by_SES.pdf",
    "figure2_ndvi_dose_response_data.csv",
    "figure4_ndvi_effect_modification_data.csv",
    "figure3_ndvi_by_zip3.csv"
]

with zipfile.ZipFile("ndvi_manuscript_outputs.zip", "w") as zipf:
    for file in output_files:
        if os.path.exists(file):
            zipf.write(file)
        else:
            print(f" Skipped missing file: {file}")

print("All available files packaged into ndvi_manuscript_outputs.zip")


# In[ ]:




