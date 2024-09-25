import requests
import pandas as pd
import csv

from lifelines import KaplanMeierFitter 
import matplotlib.pyplot as plt

fields = [
    "case_id",
    "submitter_id",
    "project.project_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.days_to_last_follow_up"
    ]

fields = ",".join(fields)

cases_endpt = "https://api.gdc.cancer.gov/cases"

# This set of filters is nested under an 'and' operator.
filters = {
    "op": "and",
    "content":[
        {
        "op": "in",
        "content":{
            "field": "cases.project.project_id",
            "value": ["TCGA-KIRP", "TCGA-KIRC", "TCGA-KICH"] #"TCGA-KIRP", "TCGA-KIRC", "TCGA-KICH"
            }
        }
    ]
}

# A POST is used, so the filter parameters can be passed directly as a Dict object.
params = {
    "filters": filters,
    "fields": fields,
    "format": "JSON",
    "size": "2000"
    }


# Send the request to GDC API
response = requests.post(cases_endpt, json=params)

# Check if the request was successful
if response.status_code == 200:
    print("Query successful")
    json_data = response.json()
else:
    print(f"Query failed with status code: {response.status_code}")
    exit()

# Extract the clinical data
cases = json_data['data']['hits']
print("The number of cases:", len(cases))

# Convert the clinical data into a pandas DataFrame
survival_data = []

for case in cases:
    survival_data.append({
        'case_id': case['case_id'],
        'submitter_id': case['submitter_id'],
        'project_id': case['project']['project_id'],
        'days_to_last_follow_up': case['diagnoses'][0].get('days_to_last_follow_up', None),
        'days_to_death': case['demographic'].get('days_to_death', None),
        'vital_status': case['demographic'].get('vital_status', None)
    })

df = pd.DataFrame(survival_data)

# Display the first few rows of the survival data
print(df.head())

df.to_csv("/Users/shangqigao/Documents/projects/Experiments/clinical/TCGA_PanKidney_survival_data.csv", index=False)
print("Survival data saved to CSV")

df = pd.read_csv("/Users/shangqigao/Documents/projects/Experiments/clinical/TCGA_PanKidney_survival_data.csv")

# Prepare the survival data
df['event'] = df['vital_status'].apply(lambda x: 1 if x == 'Dead' else 0)
df['duration'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
df = df[df['duration'].notna()]
print(df.shape)

# Fit the Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(df['duration'], event_observed=df['event'])
print(kmf.survival_function_)
print(kmf.median_survival_time_)

# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title("Kaplan-Meier Survival Curve for Pan-Kidney")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.show()