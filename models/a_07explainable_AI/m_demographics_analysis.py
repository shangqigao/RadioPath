import pandas as pd

data_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/clinical"

df_concept = pd.read_csv(f"{data_dir}/TCGA-RCC_pathological_concepts.csv")
df_clinical = pd.read_csv(f"{data_dir}/TCGA-RCC_survival_data.csv")

df_demographics = df_clinical[df_clinical['submitter_id'].isin(df_concept['index'])]

# gender
counts = df_demographics['gender'].str.lower().value_counts()
percentages = df_demographics['gender'].str.lower().value_counts(normalize=True) * 100
summary = pd.DataFrame({'Count': counts, 'Percentage': percentages.round(2)})
print(summary)

# race
counts = df_demographics['race'].str.lower().value_counts()
percentages = df_demographics['race'].str.lower().value_counts(normalize=True) * 100
summary = pd.DataFrame({'Count': counts, 'Percentage': percentages.round(2)})
print(summary)

# age
print(df_demographics['age'].describe())

# subtype
counts = df_demographics['project_id'].str.lower().value_counts()
percentages = df_demographics['project_id'].str.lower().value_counts(normalize=True) * 100
summary = pd.DataFrame({'Count': counts, 'Percentage': percentages.round(2)})
print(summary)
