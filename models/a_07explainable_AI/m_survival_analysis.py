import sys
sys.path.append('../')

import requests
import argparse
import pathlib
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from common.m_utils import mkdir, select_wsi, load_json
from lifelines import KaplanMeierFitter, CoxPHFitter 

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

def prepare_graph_properties(data_dict, prop_keys):
    properties = {}
    for subgraph, prop_dict in data_dict.items():
        if subgraph == "MUC": continue
        for k in prop_keys:
            key = f"{subgraph}.{k}"
            if prop_dict is None or len(prop_dict[k]) == 0:
                properties[key] = -1 if k == "graph_assortativity" else 0
            else:
                if len(prop_dict[k]) == 1:
                    if np.isnan(prop_dict[k][0]):
                        properties[key] = -1 if k == "graph_assortativity" else 0
                    else:
                        properties[key] = prop_dict[k][0]
                else:
                    properties[key] = np.array(prop_dict[k]).mean()
    return properties

def cox_proportional_hazard_regression(save_clinical_dir, save_properties_paths, prop_keys):
    df = pd.read_csv(f"{save_clinical_dir}/TCGA_PanKidney_survival_data.csv")
    
    # Prepare the survival data
    df['event'] = df['vital_status'].apply(lambda x: 1 if x == 'Dead' else 0)
    df['duration'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
    df = df[df['duration'].notna()]
    print("Data strcuture:", df.shape)
    
    # Prepare the graph properties
    prop_list = [load_json(p) for p in save_properties_paths]
    prop_list = [prepare_graph_properties(p, prop_keys) for p in prop_list]

    # filter graph properties 
    ids = df['submitter_id']
    names = [p.stem for p in save_properties_paths]
    matched_index, matched_i = [], []
    for index, id in ids.items():
        for i, name in enumerate(names):
            if id in name:
                matched_index.append(index)
                matched_i.append(i)
    df = df.loc[matched_index]
    df = df[['duration', 'event']]
    df = df.reset_index(drop=True)

    filtered_prop = [prop_list[i] for i in matched_i]
    df_prop = pd.DataFrame(filtered_prop)
    print("Data to concatenate:", df.shape, df_prop.shape)
    df_concat = pd.concat([df, df_prop], axis=1)
    print("New data strcuture:", df_concat.shape)
    print(list(df_concat))

    # COX regreession
    cph = CoxPHFitter(penalizer=0.01, l1_ratio=0.9)
    cph.fit(df_concat, duration_col='duration', event_col='event')
    cph.print_summary()
    cph.check_assumptions(df_concat, p_value_threshold=0.05)
    plt.figure(figsize=(10, 10))
    covariates = ["ADI.graph_assortativity", 
                  "BACK.graph_assortativity", 
                  "DEB.graph_assortativity",
                  "LYM.graph_assortativity",
                  "MUS.graph_assortativity",
                  "NORM.graph_assortativity",
                  "STR.graph_assortativity",
                  "TUM.graph_assortativity"]
    cph.plot_partial_effects_on_outcome("BACK.graph_assortativity", values=[-0.5, 0, 0.5])
    plt.savefig("a_07explainable_AI/cox_regression.jpg")
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
    wsi_dir = pathlib.Path(args.wsi_dir) / args.dataset
    all_paths = sorted(pathlib.Path(wsi_dir).rglob("*.svs"))
    excluded_wsi = ["TCGA-5P-A9KC-01Z-00-DX1", "TCGA-5P-A9KA-01Z-00-DX1", "TCGA-UZ-A9PQ-01Z-00-DX1"]
    wsi_paths = []
    for path in all_paths:
        wsi_name = f"{path}".split("/")[-1].split(".")[0]
        if wsi_name not in excluded_wsi: wsi_paths.append(path)
    logging.info("The number of selected WSIs on {}: {}".format(args.dataset, len(wsi_paths)))
    
    
    ## set save dir
    save_pathomics_dir = pathlib.Path(f"{args.save_pathomics_dir}/{args.dataset}_{args.mode}_pathomic_features/{args.feature_mode}")
    save_clinical_dir = pathlib.Path(f"{args.save_clinical_dir}")

    # request survial data by GDC API
    # project_ids = ["TCGA-KIRP", "TCGA-KIRC", "TCGA-KICH"]
    # request_survival_data(["TCGA-KIRC"], save_clinical_dir)

    # plot survival curve
    # plot_survival_curve(save_clinical_dir)

    # survival analysis
    graph_prop_paths = [save_pathomics_dir / f"{p.stem}.MST.subgraphs.properties.json" for p in wsi_paths]
    graph_properties = [
        "num_nodes", 
        "num_edges", 
        "num_components", 
        # "degree", 
        "closeness", 
        # "graph_diameter",
        "graph_assortativity",
        # "mean_neighbor_degree"
    ]
    cox_proportional_hazard_regression(
        save_clinical_dir=save_clinical_dir,
        save_properties_paths=graph_prop_paths,
        prop_keys=graph_properties
    )