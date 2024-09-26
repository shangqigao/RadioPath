import sys
sys.path.append('../')

import requests
import argparse
import pathlib
import logging

import matplotlib.pyplot as plt
import pandas as pd

from common.m_utils import mkdir, select_wsi
from lifelines import KaplanMeierFitter 

def request_survival_data(project_ids, save_dir):
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
                "value": project_ids #"TCGA-KIRP", "TCGA-KIRC", "TCGA-KICH"
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
    mkdir(save_dir)
    df.to_csv(f"{save_dir}/TCGA_PanKidney_survival_data.csv", index=False)
    print("Survival data saved to CSV")
    return

def plot_survival_curve(save_dir):
    df = pd.read_csv(f"{save_dir}/TCGA_PanKidney_survival_data.csv")

    # Prepare the survival data
    df['event'] = df['vital_status'].apply(lambda x: 1 if x == 'Dead' else 0)
    df['duration'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
    df = df[df['duration'].notna()]
    print("Data strcuture:", df.shape)

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
    plt.savefig("a_07explainable_AI/survival_curve.jpg")
    return

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', default="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/TCGA/WSI")
    parser.add_argument('--dataset', default="TCGA-RCC", type=str)
    parser.add_argument('--save_pathomics_dir', default="/home/sg2162/rds/hpc-work/Experiments/pathomics", type=str)
    parser.add_argument('--save_clinical_dir', default="/home/sg2162/rds/hpc-work/Experiments/clinical", type=str)
    parser.add_argument('--mode', default="wsi", choices=["tile", "wsi"], type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--feature_mode', default="uni", choices=["cnn", "vit", "uni", "conch"], type=str)
    parser.add_argument('--node_features', default=384, choices=[2048, 384, 1024, 35], type=int)
    parser.add_argument('--resolution', default=20, type=float)
    parser.add_argument('--units', default="power", type=str)
    args = parser.parse_args()

    ## get wsi path
    # wsi_dir = pathlib.Path(args.wsi_dir) / args.dataset
    # excluded_wsi = ["TCGA-5P-A9KC-01Z-00-DX1", "TCGA-5P-A9KA-01Z-00-DX1"]
    # wsi_paths = select_wsi(wsi_dir, excluded_wsi)
    # logging.info("The number of selected WSIs on {}: {}".format(args.dataset, len(wsi_paths)))
    
    ## set save dir
    save_pathomics_dir = pathlib.Path(f"{args.save_pathomics_dir}/{args.dataset}_{args.mode}_pathomic_features/{args.feature_mode}")
    save_clinical_dir = pathlib.Path(f"{args.save_clinical_dir}")

    # request survial data by GDC API
    # project_ids = ["TCGA-KIRP", "TCGA-KIRC", "TCGA-KICH"]
    # request_survival_data(project_ids, save_clinical_dir)

    # plot survival curve
    plot_survival_curve(save_clinical_dir)