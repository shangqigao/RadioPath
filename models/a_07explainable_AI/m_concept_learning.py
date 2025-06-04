import sys
sys.path.append('../')

import requests
import argparse
import pathlib
import logging
import warnings
import joblib
import copy
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torchbnn as bnn

from scipy.stats import zscore
from torch_geometric.loader import DataLoader
from tiatoolbox import logger
from tiatoolbox.utils.misc import save_as_json

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis, IPCRidge
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import (
    concordance_index_censored, 
    concordance_index_ipcw,
    integrated_brier_score, 
    cumulative_dynamic_auc,
    as_concordance_index_ipcw_scorer,
    as_cumulative_dynamic_auc_scorer,
    as_integrated_brier_score_scorer
)

from sklearn import set_config
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, VarianceThreshold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import balanced_accuracy_score as acc_scorer
from sklearn.metrics import average_precision_score as ap_scorer
from sklearn.metrics import f1_score as f1_scorer
from sklearn.utils import resample


from common.m_utils import mkdir, select_wsi, load_json, create_pbar, rm_n_mkdir, reset_logging, recur_find_ext, select_checkpoints

from models.a_05feature_aggregation.m_gnn_concept_learning import ConceptGraphDataset, ConceptGraphArch
from models.a_05feature_aggregation.m_gnn_concept_learning import ScalarMovingAverage, CoxSurvConceptLoss
from models.a_05feature_aggregation.m_graph_construction import visualize_pathomic_graph


def request_survival_data(project_ids, save_dir):
    fields = [
        "case_id",
        "submitter_id",
        "project.project_id",
        "demographic.gender",
        "demographic.race",
        "demographic.age_at_index",
        "demographic.vital_status",
        "demographic.days_to_death",
        "follow_ups.days_to_follow_up",
        "follow_ups.days_to_recurrence",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.ajcc_pathologic_m"
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
        follow_ups = case['follow_ups']
        days_to_follow_up = [c.get('days_to_follow_up', None) for c in follow_ups]
        days_to_follow_up = [d for d in days_to_follow_up if d is not None]
        print(f"Days to follow up: {days_to_follow_up}")
        days_to_last_follow_up = max(days_to_follow_up) if len(days_to_follow_up) > 0 else None
        days_to_recurrence = [c.get('days_to_recurrence', None) for c in follow_ups]
        days_to_recurrence = [d for d in days_to_recurrence if d is not None]
        days_to_recurrence = min(days_to_recurrence) if len(days_to_recurrence) > 0 else None
        if 'diagnoses' in list(case.keys()):
            ajcc_pathologic_stage = case['diagnoses'][0].get('ajcc_pathologic_stage', None)
            ajcc_pathologic_m = case['diagnoses'][0].get('ajcc_pathologic_m', None)
        else:
            ajcc_pathologic_stage = None
            ajcc_pathologic_m = None
        survival_data.append({
            'case_id': case['case_id'],
            'submitter_id': case['submitter_id'],
            'project_id': case['project']['project_id'],
            'days_to_last_follow_up': days_to_last_follow_up,
            'days_to_recurrence': days_to_recurrence,
            'ajcc_pathologic_stage': ajcc_pathologic_stage,
            'ajcc_pathologic_m': ajcc_pathologic_m,
            'days_to_death': case['demographic'].get('days_to_death', None),
            'vital_status': case['demographic'].get('vital_status', None),
            'gender': case['demographic'].get('gender', None),
            'race': case['demographic'].get('race', None),
            'age': case['demographic'].get('age_at_index', None)
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
    df['event'] = df['vital_status'].apply(lambda x: True if x == 'Dead' else False)
    df['duration'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
    df = df[df['duration'].notna()]
    df = df[df['ajcc_pathologic_stage'].isin(["Stage I", "Stage II"])]
    print("Data strcuture:", df.shape)

    # Fit the Kaplan-Meier estimator
    time, survival_prob, conf_int = kaplan_meier_estimator(
        df["event"], df["duration"], conf_type="log-log"
        )
    plt.step(time, survival_prob, where="post")
    plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(0, 1)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
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
                    properties[key] = np.std(prop_dict[k])
    return properties

def load_pyradiomic_properties(idx, radiomic_path, prop_keys=None):
    if prop_keys is None: 
        prop_keys = ["shape", "firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]
    data_dict = load_json(radiomic_path)
    properties = {}
    for key, value in data_dict.items():
        selected = [((k in key) and ("diagnostics" not in key)) for k in prop_keys]
        if any(selected): properties[key] = value
    return {f"{idx}": properties}

def prepare_graph_pathomics(
    idx, 
    graph_path, 
    subgraphs=["TUM", "NORM", "DEB"], 
    mode="mean"
    ):
    if subgraphs is None:
        subgraph_ids = None
    else:
        subgraph_dict = {
            "ADI": [0, 4],
            "BACK": [5, 8],
            "DEB": [9, 11],
            "LYM": [12, 16],
            "MUC": [17, 20],
            "MUS": [21, 25],
            "NORM": [26, 26],
            "STR": [27, 31],
            "TUM": [32, 34]
        }
        subgraph_ids = [subgraph_dict[k] for k in subgraphs]
    graph_dict = load_json(graph_path)
    label_path = f"{graph_path}".replace(".json", ".label.npy")
    label = np.load(label_path)
    if label.ndim == 2: label = np.argmax(label, axis=1)
    feature = np.array(graph_dict["x"])
    assert feature.ndim == 2
    if subgraph_ids is not None:
        subset = label < 0
        for ids in subgraph_ids:
            ids_subset = np.logical_and(label >= ids[0], label <= ids[1])
            subset = np.logical_or(subset, ids_subset)
        if subset.sum() < 1:
            feature = np.zeros_like(feature)
        else:
            feature = feature[subset]

    if mode == "mean":
        feat_list = np.mean(feature, axis=0).tolist()
    elif mode == "max":
        feat_list = np.max(feature, axis=0).tolist()
    elif mode == "min":
        feat_list = np.min(feature, axis=0).tolist()
    elif mode == "std":
        feat_list = np.std(feature, axis=0).tolist()
    elif mode == "kmeans":
        kmeans = KMeans(n_clusters=4)
        feat_list = kmeans.fit(feature).cluster_centers_
        feat_list = feat_list.flatten().tolist()
        
    feat_dict = {}
    for i, feat in enumerate(feat_list):
        k = f"pathomics.feature{i}"
        feat_dict[k] = feat
    return {f"{idx}": feat_dict}

def prepare_graph_radiomics(
    idx, 
    graph_path, 
    mode="mean"
    ):
    graph_dict = load_json(graph_path)
    feature = np.array(graph_dict["x"])
    assert feature.ndim == 2

    if mode == "mean":
        feat_list = np.mean(feature, axis=0).tolist()
    elif mode == "max":
        feat_list = np.max(feature, axis=0).tolist()
    elif mode == "min":
        feat_list = np.min(feature, axis=0).tolist()
    elif mode == "std":
        feat_list = np.std(feature, axis=0).tolist()
    elif mode == "kmeans":
        kmeans = KMeans(n_clusters=4)
        feat_list = kmeans.fit(feature).cluster_centers_
        feat_list = feat_list.flatten().tolist()
        
    feat_dict = {}
    for i, feat in enumerate(feat_list):
        k = f"radiomics.feature{i}"
        feat_dict[k] = feat
    return {f"{idx}": feat_dict}

def load_wsi_level_features(idx, wsi_feature_path):
    feat_list = np.array(np.load(wsi_feature_path)).squeeze().tolist()
    feat_dict = {}
    for i, feat in enumerate(feat_list):
        k = f"pathomics.feature{i}"
        feat_dict[k] = feat
    return {f"{idx}": feat_dict}

def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
    plt.subplots_adjust(left=0.2)
    plt.savefig("a_07explainable_AI/coefficients.jpg")

def matched_survival_graph(save_clinical_dir, save_graph_paths, dataset="TCGA-RCC", stages=None, survival="OS", metastasis=None):
    df = pd.read_csv(f"{save_clinical_dir}/{dataset}_survival_data.csv")
    
    # Prepare the survival data
    if survival == "OS":
        df['event'] = df['vital_status'].apply(lambda x: True if x == 'Dead' else False)
        df['duration'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
        df = df[df['duration'].notna()]
    elif survival == "DFS":
        df.fillna(float('inf'), inplace=True)
        df['event'] = df.apply(lambda row: True if row['days_to_recurrence'] < float('inf') or 
                               row['days_to_death'] < float('inf') else False, axis=1)
        df['duration'] = df[['days_to_recurrence', 'days_to_death', 'days_to_last_follow_up']].min(axis=1)
        df = df[df['duration'].notna()]
    if stages is not None:
        df = df[df['ajcc_pathologic_stage'].isin(stages)]
    if metastasis is not None:
        df = df[df['ajcc_pathologic_m'].isin(metastasis)]
    print("Survival data strcuture:", df.shape)

    # filter graph properties 
    graph_names = [pathlib.Path(p).stem for p in save_graph_paths]
    graph_ids = [f"{n}".split("-")[0:3] for n in graph_names]
    graph_ids = ["-".join(d) for d in graph_ids]
    df = df[df["submitter_id"].isin(graph_ids)]
    matched_indices = [graph_ids.index(d) for d in df["submitter_id"]]
    logging.info(f"The number of matched clinical samples are {len(matched_indices)}")
    return df, matched_indices

def matched_concepts_graph(save_concept_dir, save_graph_paths, dataset="TCGA-RCC", stages=None):
    df = pd.read_csv(f"{save_concept_dir}/{dataset}_pathological_concepts.csv")
    
    # Prepare the concept data
    # df = df.apply(lambda x: True if x == 'TRUE' else False)
    # print("Survival data strcuture:", df.shape)

    # filter graph properties 
    graph_names = [pathlib.Path(p).stem for p in save_graph_paths]
    graph_ids = [f"{n}".split("-")[0:3] for n in graph_names]
    graph_ids = ["-".join(d) for d in graph_ids]
    df = df[df["index"].isin(graph_ids)]
    matched_indices = [graph_ids.index(d) for d in df["index"]]
    logging.info(f"The number of matched conceptual samples are {len(matched_indices)}")
    df = df.drop(columns=["index"])
    return df, matched_indices

def matched_pathomics_radiomics(save_pathomics_paths, save_radiomics_paths, save_clinical_dir, dataset="TCGA-RCC", project_ids=None):
    df = pd.read_csv(f"{save_clinical_dir}/TCIA_{dataset}_mappings.csv")
    if project_ids is not None: df = df[df["Collection Name"].isin(project_ids)]

    df = df[["Subject ID", "Series ID"]]
    pathomics_names = [pathlib.Path(p).stem for p in save_pathomics_paths]
    pathomics_ids = [f"{n}".split("-")[0:3] for n in pathomics_names]
    pathomics_ids = ["-".join(d) for d in pathomics_ids]
    radiomics_names = [pathlib.Path(p).stem for p in save_radiomics_paths]
    radiomics_all_ids = [f"{n}".split(".")[0:13] for n in radiomics_names]
    radiomics_end_ids = [f"{d[-1]}".split("_")[0] for d in radiomics_all_ids]
    radiomics_ids = [id1[0:12] + [id2] for id1, id2 in zip(radiomics_all_ids, radiomics_end_ids)]
    radiomics_ids = [".".join(d) for d in radiomics_ids]

    df = df[df["Subject ID"].isin(pathomics_ids) & df["Series ID"].isin(radiomics_ids)] 
    matched_pathomics_indices, matched_radiomics_indices = [], []
    for subject_id, series_id in zip(df["Subject ID"], df["Series ID"]):
        matched_pathomics_indices.append(pathomics_ids.index(subject_id))
        matched_radiomics_indices.append(radiomics_ids.index(series_id))
    logging.info(f"The number of matched pathomic and radiomic cases are {len(matched_pathomics_indices)}")
    return matched_pathomics_indices, matched_radiomics_indices

def cox_proportional_hazard_regression(
        save_clinical_dir, 
        save_pathomics_paths,
        save_radiomics_paths,
        pathomics_keys, 
        radiomics_keys,
        l1_ratio=1.0, 
        stages=None,
        used="all", 
        n_jobs=32,
        aggregation=False,
        dataset="TCGA-RCC"
        ):
    # prepare clinical data
    df_clinical, matched_i = matched_survival_graph(save_clinical_dir, save_pathomics_paths, dataset, stages)
    num_dead = df_clinical['event'].value_counts()[True]
    num_alive = df_clinical['event'].value_counts()[False]
    df_clinical = df_clinical[['event', 'duration']].to_records(index=False)
    print("Selected survival data:", df_clinical.shape)
    print(f"Dead cases: {num_dead}, Alive cases: {num_alive}")

    # Prepare graph-based pathomics
    save_properties_paths = []
    for p in save_pathomics_paths:
        name = pathlib.Path(p).stem
        if ".WSI.features" in name:
            save_properties_paths.append(f"{p}".replace(".WSI.features.npy", ".MST.subgraphs.properties.json")) 
        elif ".MST" in name:
            save_properties_paths.append(f"{p}".replace(".MST.json", ".MST.subgraphs.properties.json"))
        else:
            raise NotImplementedError
    matched_properties_paths = [save_properties_paths[i] for i in matched_i]
    pathomics_list = [load_json(p) for p in matched_properties_paths]
    pathomics_list = [prepare_graph_properties(p, pathomics_keys) for p in pathomics_list]
    df_pathomics = pd.DataFrame(pathomics_list)
    # df_prop = OneHotEncoder().fit_transform(df_prop)
    print("Selected pathomic properties:", df_pathomics.shape)
    print(df_pathomics.head())

    # prepare radiomics
    if save_radiomics_paths is not None:
        matched_radiomics_paths = [save_radiomics_paths[i] for i in matched_i]
        if aggregation:
            dict_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(prepare_graph_radiomics)(idx, graph_path)
                for idx, graph_path in enumerate(matched_radiomics_paths)
            )
        else:
            dict_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(load_pyradiomic_properties)(idx, graph_path, radiomics_keys)
                for idx, graph_path in enumerate(matched_radiomics_paths)
            )
        radiomics_dict = {}
        for d in dict_list: radiomics_dict.update(d)
        radiomics_list = [radiomics_dict[f"{i}"] for i in range(len(matched_radiomics_paths))]
        df_radiomics = pd.DataFrame(radiomics_list)
        print("Selected radiomic properties:", df_radiomics.shape)
        print(df_radiomics.head())


    # Prepare WSI-level features
    selected_graph_paths = [save_pathomics_paths[i] for i in matched_i]
    if aggregation:
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(prepare_graph_pathomics)(idx, graph_path)
            for idx, graph_path in enumerate(selected_graph_paths)
        )
    else:
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(load_wsi_level_features)(idx, graph_path)
            for idx, graph_path in enumerate(selected_graph_paths)
        )
    feat_dict = {}
    for d in dict_list: feat_dict.update(d)
    feat_list = [feat_dict[f"{i}"] for i in range(len(selected_graph_paths))]

    df_feat = pd.DataFrame(feat_list)
    print("Selected WSI-level features:", df_feat.shape)
    print(df_feat.head())

    # Concatenate final features for regression
    if used == "all":
        df_prop = pd.concat([df_pathomics, df_radiomics, df_feat], axis=1)
    elif used == "radiopathomics":
        df_prop = pd.concat([df_pathomics, df_radiomics], axis=1)
    elif used == "radioDeepPathomics":
        df_prop = pd.concat([df_radiomics, df_feat], axis=1)
    elif used == "pathoDeepPathomics":
        df_prop = pd.concat([df_pathomics, df_feat], axis=1)
    elif used == "pathomics":
        df_prop = df_pathomics
    elif used == "radiomics":
        df_prop = df_radiomics
    elif used == "deep_pathomics":
        df_prop = df_feat
    else:
        raise NotImplementedError
    # df_prop = df_prop.apply(zscore)
    print("Selected features for regression:", df_prop.shape)
    print(df_prop.head())

    # COX regreession
    cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio="auto")
    cox_elastic_net.fit(df_prop, df_clinical)
    coefficients = pd.DataFrame(cox_elastic_net.coef_, index=df_prop.columns, columns=np.round(cox_elastic_net.alphas_, 5))
    
    plot_coefficients(coefficients, n_highlight=5)

    # choosing penalty strength by cross validation
    cox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio="auto", max_iter=1000)
    coxnet_pipe = make_pipeline(StandardScaler(), cox)
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    coxnet_pipe.fit(df_prop, df_clinical)
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    cox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=1000)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), cox),
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv=cv,
        error_score=0.5,
        n_jobs=1,
    ).fit(df_prop, df_clinical)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    print("Best CV performance:", np.max(mean))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig("a_07explainable_AI/cross_validation.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    best_coefs = pd.DataFrame(best_model.coef_, index=df_prop.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index

    _, ax = plt.subplots(figsize=(8, 6))
    non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.savefig("a_07explainable_AI/best_coefficients.jpg") 

    # perform prediction using the best params
    cox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, fit_baseline_model=True)

    coxnet_pred = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, fit_baseline_model=True))
    coxnet_pred.set_params(**gcv.best_params_)
    coxnet_pred.fit(df_prop, df_clinical)
    print("Best performance:", coxnet_pred.score(df_prop, df_clinical))
    return

def coxnet(split_idx, tr_X, tr_y, scorer, n_jobs, l1_ratio=0.9, min_ratio=0.1):
    # COX regreession
    print("Selecting the best regularization parameter...")
    cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=min_ratio)
    cox_elastic_net.fit(tr_X, tr_y)
    coefficients = pd.DataFrame(cox_elastic_net.coef_, index=tr_X.columns, columns=np.round(cox_elastic_net.alphas_, 5))
    
    plot_coefficients(coefficients, n_highlight=5)

    # choosing penalty strength by cross validation
    cox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=min_ratio, max_iter=10000)
    coxnet_pipe = make_pipeline(StandardScaler(), cox)
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    coxnet_pipe.fit(tr_X, tr_y)
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=10000, fit_baseline_model=True)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__alphas": [[v] for v in estimated_alphas]}
    elif scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__alphas": [[v] for v in estimated_alphas]}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__alphas": [[v] for v in estimated_alphas]}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__alphas": [[v] for v in estimated_alphas]}
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        alphas = cv_results.param_model__alphas.map(lambda x: x[0])
    else:
        alphas = cv_results.param_model__estimator__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(score_name)
    ax.set_xlabel("alpha")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__alphas"][0], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__alphas"][0], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"a_07explainable_AI/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]
    if scorer == "cindex":
        best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["coefficient"])
    else:
        best_coefs = pd.DataFrame(best_model.estimator_.coef_, index=tr_X.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index

    _, ax = plt.subplots(figsize=(8, 6))
    non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"a_07explainable_AI/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def rsf(split_idx, tr_X, tr_y, scorer, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    model = RandomSurvivalForest(max_depth=2, random_state=1)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    else:
        raise NotImplementedError

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        depths = cv_results.param_model__max_depth
    else:
        depths = cv_results.param_model__estimator__max_depth
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depths, mean)
    ax.fill_between(depths, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("linear")
    ax.set_ylabel(score_name)
    ax.set_xlabel("max depth")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__max_depth"], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__max_depth"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"a_07explainable_AI/cross_validation_fold{split_idx}.jpg")

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def gradientboosting(split_idx, tr_X, tr_y, scorer, n_jobs, loss="coxph"):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = GradientBoostingSurvivalAnalysis(loss=loss, max_depth=2, random_state=1)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    else:
        raise NotImplementedError

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        depths = cv_results.param_model__max_depth
    else:
        depths = cv_results.param_model__estimator__max_depth
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depths, mean)
    ax.fill_between(depths, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("linear")
    ax.set_ylabel(score_name)
    ax.set_xlabel("max depth")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__max_depth"], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__max_depth"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"a_07explainable_AI/cross_validation_fold{split_idx}.jpg")

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe


def coxph(split_idx, tr_X, tr_y, scorer, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    model = CoxPHSurvivalAnalysis(alpha=1e-2)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__alpha": 10.0 ** np.arange(-2, 5)}
    elif scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__alpha": 10.0 ** np.arange(-2, 5)}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 10.0 ** np.arange(-2, 5)}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 10.0 ** np.arange(-2, 5)}
    else:
        raise NotImplementedError

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        alphas = cv_results.param_model__alpha
    else:
        alphas = cv_results.param_model__estimator__alpha
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(score_name)
    ax.set_xlabel("alpha")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__alpha"], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__alpha"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"a_07explainable_AI/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]
    if scorer == "cindex":
        best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["coefficient"])
    else:
        best_coefs = pd.DataFrame(best_model.estimator_.coef_, index=tr_X.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index
    top10 = coef_order[:10]

    _, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams.update({'font.size': 18})
    non_zero_coefs.loc[top10].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.subplots_adjust(left=0.6)
    plt.savefig(f"a_07explainable_AI/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def ipcridge(split_idx, tr_X, tr_y, scorer, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = IPCRidge(alpha=1, random_state=1)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__alpha": 2.0 ** np.arange(0, 26, 2)}
    if scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(0, 26, 2)}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(0, 26, 2)}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(0, 26, 2)}
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        alphas = cv_results.param_model__alpha
    else:
        alphas = cv_results.param_model__estimator__alpha
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(score_name)
    ax.set_xlabel("alpha")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__alpha"], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__alpha"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"a_07explainable_AI/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]
    if scorer == "cindex":
        best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["coefficient"])
    else:
        best_coefs = pd.DataFrame(best_model.estimator_.coef_, index=tr_X.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index
    top10 = coef_order[:10]

    _, ax = plt.subplots(figsize=(8, 6))
    non_zero_coefs.loc[top10].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"a_07explainable_AI/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def fastsvm(split_idx, tr_X, tr_y, scorer, n_jobs, rank_ratio=1):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = FastSurvivalSVM(alpha=1, rank_ratio=rank_ratio)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__alpha": 2.0 ** np.arange(-26, 0, 2)}
    if scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(-26, 0, 2)}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(-26, 0, 2)}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(-26, 0, 2)}
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        alphas = cv_results.param_model__alpha
    else:
        alphas = cv_results.param_model__estimator__alpha
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(score_name)
    ax.set_xlabel("alpha")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__alpha"], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__alpha"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"a_07explainable_AI/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]
    if scorer == "cindex":
        best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["coefficient"])
    else:
        best_coefs = pd.DataFrame(best_model.estimator_.coef_, index=tr_X.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index
    top10 = coef_order[:10]

    _, ax = plt.subplots(figsize=(8, 6))
    non_zero_coefs.loc[top10].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"a_07explainable_AI/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def survival(
        split_path,
        concepts,
        pathomics_keys=None,
        pathomics_aggregation=False,
        pathomics_aggregated_mode=None,
        used="all", 
        n_jobs=32,
        model="CoxPH",
        scorer="cindex",
        feature_selection=True,
        n_bootstraps=100
        ):
    splits = joblib.load(split_path)
    predict_results = {}
    risk_results = {"risk": [], "event": [], "duration": [], "gender": [], "race": []}
    for split_idx, split in enumerate(splits):
        print(f"Performing cross-validation on fold {split_idx}...")
        data_tr, data_va, data_te = split["train"], split["valid"], split["test"]
        data_tr = data_tr + data_va

        tr_y = np.array([p[1]["label"] for p in data_tr])
        tr_y = pd.DataFrame({'event': tr_y[:, 1].astype(bool), 'duration': tr_y[:, 0]})
        tr_y = tr_y.to_records(index=False)
        te_y = np.array([p[1]["label"] for p in data_te])
        te_y = pd.DataFrame({'event': te_y[:, 1].astype(bool), 'duration': te_y[:, 0]})
        te_y = te_y.to_records(index=False)

        gender = [p[1]["demographic"][0] for p in data_tr]
        race = [p[1]["demographic"][1] for p in data_tr]
        tr_demographic = pd.DataFrame({'gender': gender, 'race': race})
        gender = [p[1]["demographic"][0] for p in data_te]
        race = [p[1]["demographic"][1] for p in data_te]
        te_demographic = pd.DataFrame({'gender': gender, 'race': race})
        
        # prepare concepts
        concept_tr_X = np.array([p[1]["concept"] for p in data_tr])
        concept_tr_X = pd.DataFrame({c: concept_tr_X[:, i] for i, c in enumerate(concepts)})
        print("Selected training concepts:", concept_tr_X.shape)
        print(concept_tr_X.head())
        concept_te_X = np.array([p[1]["concept"] for p in data_te])
        concept_te_X = pd.DataFrame({c: concept_te_X[:, i] for i, c in enumerate(concepts)})
        print("Selected testing concepts:", concept_te_X.shape)
        print(concept_te_X.head())

        # Prepare pathomics
        if pathomics_aggregated_mode in ["ABMIL", "CBM"]:
            pathomics_tr_paths = []
            for p in data_tr:
                tr_path = pathlib.Path(p[0]["pathomics"])
                tr_dir = tr_path.parents[0] / f"CL_{pathomics_aggregated_mode}_MICCAI" / f"0{split_idx}"
                tr_name = tr_path.name.replace(".json", ".npy")
                pathomics_tr_paths.append(tr_dir / tr_name)
        else:
            pathomics_tr_paths = [p[0]["pathomics"] for p in data_tr]

        if pathomics_aggregation:
            dict_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(prepare_graph_pathomics)(idx, graph_path, pathomics_keys)
                for idx, graph_path in enumerate(pathomics_tr_paths)
            )
        else:
            dict_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(load_wsi_level_features)(idx, graph_path)
                for idx, graph_path in enumerate(pathomics_tr_paths)
            )
        pathomics_dict = {}
        for d in dict_list: pathomics_dict.update(d)
        pathomics_tr_X = [pathomics_dict[f"{i}"] for i in range(len(pathomics_tr_paths))]
        pathomics_tr_X = pd.DataFrame(pathomics_tr_X)
        if pathomics_aggregated_mode == "CBM":
            if len(pathomics_tr_X.columns) == len(concepts): 
                pathomics_tr_X.columns = concepts
        print("Selected training pathomics:", pathomics_tr_X.shape)
        print(pathomics_tr_X.head())

        if pathomics_aggregated_mode in ["ABMIL", "CBM"]:
            pathomics_te_paths = []
            for p in data_te:
                te_path = pathlib.Path(p[0]["pathomics"])
                te_dir = te_path.parents[0] / f"CL_{pathomics_aggregated_mode}_MICCAI" / f"0{split_idx}"
                te_name = te_path.name.replace(".json", ".npy")
                pathomics_te_paths.append(te_dir / te_name)
        else:
            pathomics_te_paths = [p[0]["pathomics"] for p in data_te]

        if pathomics_aggregation:
            dict_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(prepare_graph_pathomics)(idx, graph_path, pathomics_keys)
                for idx, graph_path in enumerate(pathomics_te_paths)
            )
        else:
            dict_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(load_wsi_level_features)(idx, graph_path)
                for idx, graph_path in enumerate(pathomics_te_paths)
            )
        pathomics_dict = {}
        for d in dict_list: pathomics_dict.update(d)
        pathomics_te_X = [pathomics_dict[f"{i}"] for i in range(len(pathomics_te_paths))]
        pathomics_te_X = pd.DataFrame(pathomics_te_X)
        if pathomics_aggregated_mode == "CBM": 
            if len(pathomics_te_X.columns) == len(concepts):
                pathomics_te_X.columns = concepts
        print("Selected testing pathomics:", pathomics_te_X.shape)
        print(pathomics_te_X.head())

        # Concatenate multi-omics if required
        if used == "pathomics":
            tr_X, te_X = pathomics_tr_X, pathomics_te_X
        elif used == "concepts":
            tr_X, te_X = concept_tr_X, concept_te_X
        else:
            raise NotImplementedError
        # df_prop = df_prop.apply(zscore)
        print("Selected training omics:", tr_X.shape)
        print(tr_X.head())
        print("Selected testing omics:", te_X.shape)
        print(te_X.head())

        # feature selection
        if feature_selection:
            print("Selecting features...")
            # print(tr_X['wavelet-HHL_glcm_ClusterProminence'][tr_y["event"]].mean())
            # print(tr_X['wavelet-HHL_glcm_ClusterProminence'][tr_y["event"]].var())
            # print(tr_X['wavelet-HHL_glcm_ClusterProminence'][~tr_y["event"]].var())
            selector = VarianceThreshold(threshold=1e-4)
            selector.fit(tr_X[tr_y["event"]])
            selected_names = selector.get_feature_names_out().tolist()
            # selector = VarianceThreshold(threshold=1e4)
            # selector.fit(tr_X[~tr_y["event"]])
            # removed_names = selector.get_feature_names_out().tolist()
            # selected_names = [n for n in selected_names if n not in removed_names]
            num_removed = len(tr_X.columns) - len(selected_names)
            print(f"Removing {num_removed} low-variance features...")
            tr_X = tr_X[selected_names]
            te_X = te_X[selected_names]
            print("Selecting univariate feature...")
            univariate_results = []
            for name in list(tr_X.columns):
                cph = CoxPHFitter()
                df = pd.DataFrame(
                    {
                        "duration": tr_y["duration"], 
                        "event": tr_y["event"],
                        name: tr_X[name] 
                    }
                )
                cph.fit(df, "duration", "event")
                summary = cph.summary
                univariate_results.append({
                    'name': name,
                    'coef': summary['coef'].values[0],
                    'HR': summary['exp(coef)'].values[0],
                    'p_value': summary['p'].values[0],
                    'CI_low': summary['exp(coef) lower 95%'].values[0],
                    'CI_high': summary['exp(coef) upper 95%'].values[0]
                })
            results_df = pd.DataFrame(univariate_results)
            selected_names = results_df[results_df['p_value'] < 0.2]['name'].tolist()
            print(f"Selected features: {len(selected_names)}")
            tr_X = tr_X[selected_names]
            te_X = te_X[selected_names]

        # model selection
        print("Selecting survival model...")
        if model == "Coxnet":
            predictor = coxnet(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model == "RSF":
            predictor = rsf(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model == "CoxPH":
            predictor = coxph(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model == "GradientBoost":
            predictor = gradientboosting(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model == "IPCRidge":
            predictor = ipcridge(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model == "FastSVM":
            predictor = fastsvm(split_idx, tr_X, tr_y, scorer, n_jobs, rank_ratio=1)

        # bootstrapping
        if n_bootstraps > 0:
            print("Bootstrapping...")
            stable_coefs = np.zeros(len(selected_names))
            for _ in range(n_bootstraps):
                tr_x_s, tr_y_s = resample(tr_X, tr_y)
                predictor.fit(tr_x_s, tr_y_s)
                if scorer == "cindex":
                    stable_coefs += (predictor.named_steps["model"].coef_ != 0).astype(int)
                else:
                    stable_coefs += (predictor.named_steps["model"].estimator_.coef_ != 0).astype(int)
            stable_coefs = stable_coefs / n_bootstraps
            final_coefs = np.where(stable_coefs > 0.8)[0]
            stable_names = [selected_names[i] for i in final_coefs.tolist()]
            tr_X = tr_X[stable_names]
            te_X = te_X[stable_names]
            predictor.fit(tr_X, tr_y)

        risk_scores = predictor.predict(te_X)
        risk_results["risk"] += risk_scores.tolist()
        risk_results["event"] += te_y["event"].astype(int).tolist()
        risk_results["duration"] += te_y["duration"].tolist()
        risk_results["gender"] += te_demographic["gender"].tolist()
        risk_results["race"] += te_demographic["race"].tolist()
        C_index = concordance_index_censored(te_y["event"], te_y["duration"], risk_scores)[0]
        C_index_ipcw = concordance_index_ipcw(tr_y, te_y, risk_scores)[0]

        lower, upper = np.percentile(te_y["duration"], [10, 90])
        times = np.arange(lower, upper + 1, 7)
        auc, mean_auc = cumulative_dynamic_auc(tr_y, te_y, risk_scores, times)
        if hasattr(predictor, "predict_survival_function"):
            survs = predictor.predict_survival_function(te_X)
            preds = np.asarray([[fn(t) for t in times] for fn in survs])
            IBS = integrated_brier_score(tr_y, te_y, preds, times)
        else:
            IBS = 0
        scores_dict = {
            "C-index": C_index,
            "C-index-IPCW": C_index_ipcw,
            "Mean AUC": mean_auc,
            "IBS": IBS
        }

        fig, ax = plt.subplots(figsize=(9, 6))
        plt.rcParams.update({'font.size': 18})
        ax.plot(times, auc)
        ax.set_xscale("linear")
        ax.set_ylabel("time-dependent AUC")
        ax.set_xlabel("days from enrollment")
        ax.axhline(mean_auc, linestyle="--")
        ax.grid(True)
        plt.savefig(f"a_07explainable_AI/AUC_fold{split_idx}.jpg") 

        print(f"Updating regression results on fold {split_idx}")
        predict_results.update({f"Fold {split_idx}": scores_dict})
    print(predict_results)
    for k in scores_dict.keys():
        arr = np.array([v[k] for v in predict_results.values()])
        print(f"CV {k} mean+std", arr.mean(), arr.std())

    # plot survival curve on risk
    pd_risk = pd.DataFrame(risk_results)
    mean_risk = pd_risk["risk"].mean()
    dem = pd_risk["risk"] > mean_risk
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    kmf1 = KaplanMeierFitter()
    kmf1.fit(pd_risk["duration"][dem], event_observed=pd_risk["event"][dem], label="High risk")
    kmf1.plot_survival_function(ax=ax)

    kmf2 = KaplanMeierFitter()
    kmf2.fit(pd_risk["duration"][~dem], event_observed=pd_risk["event"][~dem], label="Low risk")
    # kmf.plot_survival_function(ax=ax)
    kmf2.plot_survival_function(ax=ax)
    add_at_risk_counts(kmf1, kmf2, ax=ax)
    plt.tight_layout()

    # logrank test
    test_results = logrank_test(
        pd_risk["duration"][dem], pd_risk["duration"][~dem], 
        pd_risk["event"][dem], pd_risk["event"][~dem], 
        alpha=.99
    )
    test_results.print_summary()
    pvalue = test_results.p_value
    print(f"Overall p-value: {pvalue}")
    # ax.text(1, 2, f'p-value: {pvalue}', fontsize=12, color='black', ha='left', va='bottom')
    ax.set_ylabel("Survival Probability")
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig("a_07explainable_AI/CL_survival_curve.png")

    # plot survival curve on race
    pd_white = pd_risk[pd_risk['race'] == 'white']
    print(f"Totally {len(pd_white)} white patients")
    pd_black = pd_risk[pd_risk['race'] == 'black or african american']
    print(f"Totally {len(pd_black)} black patients")
    pd_asian = pd_risk[pd_risk['race'] == 'asian']
    print(f"Totally {len(pd_asian)} asian patients")
    dem_white = pd_white["risk"] > mean_risk
    dem_black = pd_black["risk"] > mean_risk
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    kmf1 = KaplanMeierFitter()
    kmf1.fit(pd_white["duration"][dem_white], event_observed=pd_white["event"][dem_white], label="White High risk")
    kmf1.plot_survival_function(ax=ax)

    kmf2 = KaplanMeierFitter()
    kmf2.fit(pd_black["duration"][dem_black], event_observed=pd_black["event"][dem_black], label="Black High risk")
    kmf2.plot_survival_function(ax=ax)
    add_at_risk_counts(kmf1, kmf2, ax=ax)
    plt.tight_layout()

    # logrank test
    test_results = logrank_test(
        pd_white["duration"][dem_white], pd_black["duration"][dem_black], 
        pd_white["event"][dem_white], pd_black["event"][dem_black], 
        alpha=.99
    )
    test_results.print_summary()
    pvalue = test_results.p_value
    print(f"Race high-risk p-value: {pvalue}")
    ax.text(1, 2, f'p-value: {pvalue}', fontsize=12, color='black', ha='left', va='bottom')
    ax.set_ylabel("Survival Probability")
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig("a_07explainable_AI/CL_survival_race_high_risk.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    kmf1 = KaplanMeierFitter()
    kmf1.fit(pd_white["duration"][~dem_white], event_observed=pd_white["event"][~dem_white], label="White Low risk")
    kmf1.plot_survival_function(ax=ax)

    kmf2 = KaplanMeierFitter()
    kmf2.fit(pd_black["duration"][~dem_black], event_observed=pd_black["event"][~dem_black], label="Black Low risk")
    kmf2.plot_survival_function(ax=ax)
    add_at_risk_counts(kmf1, kmf2, ax=ax)
    plt.tight_layout()

    # logrank test
    test_results = logrank_test(
        pd_white["duration"][~dem_white], pd_black["duration"][~dem_black], 
        pd_white["event"][~dem_white], pd_black["event"][~dem_black], 
        alpha=.99
    )
    test_results.print_summary()
    pvalue = test_results.p_value
    print(f"Race low-risk p-value: {pvalue}")
    ax.text(1, 2, f'p-value: {pvalue}', fontsize=12, color='black', ha='left', va='bottom')
    ax.set_ylabel("Survival Probability")
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig("a_07explainable_AI/CL_survival_race_low_risk.png")

    # plot survival curve on gender
    pd_female = pd_risk[pd_risk['gender'] == 'female']
    print(f"Totally {len(pd_female)} female patients")
    pd_male = pd_risk[pd_risk['gender'] == 'male']
    print(f"Totally {len(pd_male)} male patients")
    dem_famale = pd_female["risk"] > mean_risk
    dem_male = pd_male["risk"] > mean_risk
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    kmf1 = KaplanMeierFitter()
    kmf1.fit(pd_female["duration"][dem_famale], event_observed=pd_female["event"][dem_famale], label="Female High risk")
    kmf1.plot_survival_function(ax=ax)

    kmf2 = KaplanMeierFitter()
    kmf2.fit(pd_male["duration"][dem_male], event_observed=pd_male["event"][dem_male], label="Male High risk")
    kmf2.plot_survival_function(ax=ax)
    add_at_risk_counts(kmf1, kmf2, ax=ax)
    plt.tight_layout()

    # logrank test
    test_results = logrank_test(
        pd_female["duration"][dem_famale], pd_male["duration"][dem_male], 
        pd_female["event"][dem_famale], pd_male["event"][dem_male], 
        alpha=.99
    )
    test_results.print_summary()
    pvalue = test_results.p_value
    print(f"Gender high-risk p-value: {pvalue}")
    ax.text(1, 2, f'p-value: {pvalue}', fontsize=12, color='black', ha='left', va='bottom')
    ax.set_ylabel("Survival Probability")
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig("a_07explainable_AI/CL_survival_gender_high_risk.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    kmf1 = KaplanMeierFitter()
    kmf1.fit(pd_female["duration"][~dem_famale], event_observed=pd_female["event"][~dem_famale], label="Female Low risk")
    kmf1.plot_survival_function(ax=ax)

    kmf2 = KaplanMeierFitter()
    kmf2.fit(pd_male["duration"][~dem_male], event_observed=pd_male["event"][~dem_male], label="Male Low risk")
    kmf2.plot_survival_function(ax=ax)
    add_at_risk_counts(kmf1, kmf2, ax=ax)
    plt.tight_layout()

    # logrank test
    test_results = logrank_test(
        pd_female["duration"][~dem_famale], pd_male["duration"][~dem_male], 
        pd_female["event"][~dem_famale], pd_male["event"][~dem_male], 
        alpha=.99
    )
    test_results.print_summary()
    pvalue = test_results.p_value
    print(f"Gender low-risk p-value: {pvalue}")
    ax.text(1, 2, f'p-value: {pvalue}', fontsize=12, color='black', ha='left', va='bottom')
    ax.set_ylabel("Survival Probability")
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig("a_07explainable_AI/CL_survival_gender_low_risk.png")
    return

def generate_data_split(
        x: list,
        y: list,
        train: float,
        valid: float,
        test: float,
        num_folds: int,
        seed: int = 5,
):
    """Helper to generate splits
    Args:
        x (list): a list of image paths
        y (list): a list of annotation paths
        train (float): ratio of training samples
        valid (float): ratio of validating samples
        test (float): ratio of testing samples
        num_folds (int): number of folds for cross-validation
        seed (int): random seed
    Returns:
        splits (list): a list of folds, each fold consists of train, valid, and test splits
    """
    assert train + valid + test - 1.0 < 1.0e-10, "Ratios must sum to 1.0"

    outer_splitter = StratifiedShuffleSplit(
        n_splits=num_folds,
        train_size=train + valid,
        random_state=seed,
    )
    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train / (train + valid),
        random_state=seed,
    )

    l = np.array([i["label"][1] for i in y])

    splits = []
    for train_valid_idx, test_idx in outer_splitter.split(x, l):
        test_x = [x[idx] for idx in test_idx]
        test_y = [y[idx] for idx in test_idx]
        x_ = [x[idx] for idx in train_valid_idx]
        y_ = [y[idx] for idx in train_valid_idx]
        l_ = [l[idx] for idx in train_valid_idx]

        if valid > 0:
            train_idx, valid_idx = next(iter(inner_splitter.split(x_, l_)))
            valid_x = [x_[idx] for idx in valid_idx]
            valid_y = [y_[idx] for idx in valid_idx]
            train_x = [x_[idx] for idx in train_idx]
            train_y = [y_[idx] for idx in train_idx]
        else:
            train_x, train_y = x_, y_

        train_x_list = []
        for i in train_x: train_x_list + list(i.values())
        test_x_list = []
        for i in test_x: test_x_list + list(i.values())
        if valid > 0:
            valid_x_list = []
            for i in valid_x: valid_x_list + list(i.values())
            assert len(set(train_x_list).intersection(set(valid_x_list))) == 0
            assert len(set(valid_x_list).intersection(set(test_x_list))) == 0
        else:
            assert len(set(train_x_list).intersection(set(test_x_list))) == 0

        if valid > 0:
            splits.append(
                {
                    "train": list(zip(train_x, train_y)),
                    "valid": list(zip(valid_x, valid_y)),
                    "test": list(zip(test_x, test_y)),
                }
            )
        else:
            splits.append(
                {
                    "train": list(zip(train_x, train_y)),
                    "test": list(zip(test_x, test_y)),
                }
            )
    return splits

def run_once(
        dataset_dict,
        num_epochs,
        save_dir,
        on_gpu=True,
        preproc_func=None,
        pretrained=None,
        loader_kwargs=None,
        arch_kwargs=None,
        optim_kwargs=None,
        BayesGNN=False,
        data_types=["pathomics"],
        concept_weight=None,
        use_histopath=False
):
    """running the inference or training loop once"""
    if loader_kwargs is None:
        loader_kwargs = {}

    if arch_kwargs is None:
        arch_kwargs = {}

    if optim_kwargs is None:
        optim_kwargs = {}

    model = ConceptGraphArch(**arch_kwargs)
    if pretrained is not None:
        logging.info(f"loading {pretrained}...")
        model.load(*pretrained, on_gpu)
    if on_gpu:
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    loss = CoxSurvConceptLoss(tau=0.1, concept_weight=concept_weight)
    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, **optim_kwargs)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        ds = ConceptGraphDataset(
            subset, 
            mode=subset_name, 
            preproc=preproc_func,
            data_types=data_types,
            use_histopath=use_histopath
        )
        loader_dict[subset_name] = DataLoader(
            ds,
            drop_last=subset_name == "train",
            shuffle=subset_name == "train",
            **_loader_kwargs,
        )

    best_score = 0
    for epoch in range(num_epochs):
        logger.info("EPOCH: %03d", epoch)
        for loader_name, loader in loader_dict.items():
            step_output = []
            ema = ScalarMovingAverage()
            pbar = create_pbar(loader_name, len(loader))
            for step, batch_data in enumerate(loader):
                if loader_name == "train":
                    output = model.train_batch(model, batch_data, on_gpu, loss, optimizer)
                    ema({"loss": output[0]})
                    pbar.postfix[1]["step"] = step
                    pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
                else:
                    output = model.infer_batch(model, batch_data, on_gpu)
                    batch_size = output[0].shape[0]
                    # print(batch_size, [v.shape for v in output])
                    output = [np.split(v, batch_size, axis=0) for v in output]
                    output = list(zip(*output))
                    step_output += output
                pbar.update()
            pbar.close()

            logging_dict = {}
            if loader_name == "train":
                for val_name, val in ema.tracking_dict.items():
                    logging_dict[f"train-EMA-{val_name}"] = val
            elif "infer" in loader_name and any(v in loader_name for v in ["train", "valid"]):
                output = list(zip(*step_output))
                logit, true = output[0], output[1]
                logit = np.array(logit).squeeze()
                hazard = np.exp(logit)
                true = np.array(true).squeeze()
                event_status = true[:, 1] > 0
                event_time = true[:, 0]

                cindex = concordance_index_censored(event_status, event_time, hazard)[0]
                logging_dict[f"{loader_name}-Cindex"] = cindex

                if arch_kwargs["aggregation"] == "CBM":
                    concept_logit, concept_true = output[2], output[3]
                    concept_logit = np.array(concept_logit).squeeze()
                    concept_true = np.array(concept_true).squeeze().astype(np.int8)
                    if "infer-train" in loader_name:
                        for i in range(concept_logit.shape[1]):
                            scaler = PlattScaling(solver="liblinear")
                            logit = concept_logit[:, i:i+1]
                            true = concept_true[:, i]
                            scaler.fit(logit, true)
                            model.aux_model[f"scaler{i+1}"] = scaler

                    if len(model.aux_model) == 0:
                        concept_prob = 1 / (1 + np.exp(-concept_logit))
                    else:
                        concept_prob = []
                        for i in range(concept_logit.shape[1]):
                            scaler = model.aux_model[f"scaler{i+1}"]
                            logit = concept_logit[:, i:i+1]
                            prob = scaler.predict_proba(logit)[:, 1:2]
                            concept_prob.append(prob)
                        concept_prob = np.concatenate(concept_prob, axis=1)
                    
                    concept_label = (concept_prob > 0.5).astype(np.int8)

                    acc_list = []
                    for i in range(concept_true.shape[1]):
                        acc_list.append(acc_scorer(concept_true[:, i], concept_label[:, i]))
                    acc_list = sorted(acc_list)
                    logging_dict[f"{loader_name}-top10_acc"] = sum(acc_list[-10:]) / 10
                    logging_dict[f"{loader_name}-mean_acc"] = sum(acc_list) / len(acc_list)
                    logging_dict[f"{loader_name}-bottom10_acc"] = sum(acc_list[:10]) / 10

                    concept_sum = concept_true.sum(axis=0)
                    selected = (concept_sum > 0) & (concept_sum < len(concept_true))
                    concept_true = concept_true[:, selected]
                    concept_prob = concept_prob[:, selected]
                    auroc_list = []
                    for i in range(concept_true.shape[1]):
                        auroc_list.append(auroc_scorer(concept_true[:, i], concept_prob[:, i]))
                    auroc_list = sorted(auroc_list)
                    logging_dict[f"{loader_name}-top10_auroc"] = sum(auroc_list[-10:]) / 10
                    logging_dict[f"{loader_name}-mean_auroc"] = sum(auroc_list) / len(auroc_list)
                    logging_dict[f"{loader_name}-bottom10_auroc"] = sum(auroc_list[:10]) / 10

                    # save best model based on a metric
                    mean_auroc = sum(auroc_list) / len(auroc_list)
                    if "valid-A" in loader_name and mean_auroc > best_score: 
                        best_score = mean_auroc
                        model.save(
                            f"{save_dir}/best_model.weights.pth",
                            f"{save_dir}/best_model.aux.dat",
                            )

                logging_dict[f"{loader_name}-raw-logit"] = logit
                logging_dict[f"{loader_name}-raw-true"] = true

            for val_name, val in logging_dict.items():
                if "raw" not in val_name:
                    logging.info("%s: %0.5f\n", val_name, val)
            
            if "train" not in loader_dict:
                continue

            if (epoch + 1) % 10 == 0:
                new_stats = {}
                if (save_dir / "stats.json").exists():
                    old_stats = load_json(f"{save_dir}/stats.json")
                    save_as_json(old_stats, f"{save_dir}/stats.old.json", exist_ok=True)
                    new_stats = copy.deepcopy(old_stats)
                    new_stats = {int(k): v for k, v in new_stats.items()}

                old_epoch_stats = {}
                if epoch in new_stats:
                    old_epoch_stats = new_stats[epoch]
                old_epoch_stats.update(logging_dict)
                new_stats[epoch] = old_epoch_stats
                save_as_json(new_stats, f"{save_dir}/stats.json", exist_ok=True)
                model.save(
                    f"{save_dir}/epoch={epoch:03d}.weights.pth",
                    f"{save_dir}/epoch={epoch:03d}.aux.dat",
                )
        lr_scheduler.step()
    
    return step_output


def training(
        num_epochs,
        split_path,
        scaler_path,
        num_node_features,
        num_concepts,
        model_dir,
        conv="GCNConv",
        n_works=32,
        batch_size=32,
        dropout=0,
        BayesGNN=False,
        omic_keys=["pathomics"],
        aggregation="CBM",
        concept_weight=None,
        use_histopath=False
):
    """train node classification neural networks
    Args:
        num_epochs (int): the number of epochs for training
        split_path (str): the path of storing data splits
        scaler_path (str): the path of storing data normalization
        num_node_features (int): the dimension of node feature
        model_dir (str): directory of saving models
    """
    splits = joblib.load(split_path)
    node_scalers = [joblib.load(scaler_path[k]) for k in omic_keys] 
    transform_dict = {k: s.transform for k, s in zip(omic_keys, node_scalers)}
    
    loader_kwargs = {
        "num_workers": n_works, 
        "batch_size": batch_size,
    }
    if use_histopath: 
        num_node_features["pathomics"] = num_node_features["pathomics"] + 35
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_concept": num_concepts,
        "dim_target": 1,
        "layers": [384, 256, 128, 64], # [16, 16, 8]
        "dropout": dropout,  #0.5
        "conv": conv,
        "keys": omic_keys,
        "aggregation": aggregation
    }
    if concept_weight is None: 
        CL = "CL_unweighted"
    else:
        CL = "CL_weighted"
    if use_histopath:
        HP = "Histopath"
    else:
        HP = ""
    if BayesGNN:
        model_dir = model_dir / f"{CL}_Bayes_Survival_Prediction_{conv}_{aggregation}_{HP}"
    else:
        model_dir = model_dir / f"28{CL}_calibrated_DFS_Prediction_{conv}_{aggregation}_{HP}"
    optim_kwargs = {
        "lr": 3e-4,
        "weight_decay": 1.0e-5,  # 1.0e-4
    }
    for split_idx, split in enumerate(splits):
        new_split = {
            "train": split["train"],
            "infer-train": split["train"],
            "infer-valid-A": split["valid"],
            "infer-valid-B": split["test"],
        }
        split_save_dir = pathlib.Path(f"{model_dir}/{split_idx:02d}/")
        rm_n_mkdir(split_save_dir)
        reset_logging(split_save_dir)
        run_once(
            new_split,
            num_epochs,
            save_dir=split_save_dir,
            arch_kwargs=arch_kwargs,
            loader_kwargs=loader_kwargs,
            optim_kwargs=optim_kwargs,
            preproc_func=transform_dict,
            BayesGNN=BayesGNN,
            data_types=omic_keys,
            concept_weight=concept_weight,
            use_histopath=use_histopath
        )
    return

def inference(
        split_path,
        scaler_path,
        num_node_features,
        num_concepts,
        pretrained_dir,
        conv="GCNConv",
        n_works=32,
        batch_size=32,
        dropout=0,
        BayesGNN=False,
        omic_keys=["pathomics"],
        aggregation="CBM",
        use_histopath=False,
        save_features=False
):
    """survival prediction
    """
    # torch.multiprocessing.set_start_method('spawn', force=True)
    splits = joblib.load(split_path)
    node_scalers = [joblib.load(scaler_path[k]) for k in omic_keys] 
    transform_dict = {k: s.transform for k, s in zip(omic_keys, node_scalers)}
    
    loader_kwargs = {
        "num_workers": n_works, 
        "batch_size": batch_size,
        "persistent_workers": True,
        "pin_memory": True,
    }
    if use_histopath: 
        num_node_features["pathomics"] = num_node_features["pathomics"] + 35
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_concept": num_concepts,
        "dim_target": 1,
        "layers": [384, 256, 128, 64], # [16, 16, 8]
        "dropout": dropout,  #0.5
        "conv": conv,
        "keys": omic_keys,
        "aggregation": aggregation
    }
    pretrained_dir = pretrained_dir / f"28CL_unweighted_calibrated_DFS_Prediction_GATConv_{aggregation}_"
    cum_stats = []
    cum_prob, cum_label, cum_true = [], [], []
    for split_idx, split in enumerate(splits):
        # if split_idx < 2: continue
        tr_samples = splits[split_idx]["train"] + splits[split_idx]["valid"]
        if save_features:
            all_samples = tr_samples + splits[split_idx]["test"]
        else:
            all_samples = split["test"]
        new_split = {"infer": [v[0] for v in all_samples]}

        # stat_files = recur_find_ext(f"{pretrained_dir}/{split_idx:02d}/", [".json"])
        # stat_files = [v for v in stat_files if ".old.json" not in v]
        # assert len(stat_files) == 1
        # chkpts, _ = select_checkpoints(
        #     stat_files[0],
        #     top_k=1,
        #     metric="infer-valid-A-auroc",
        # )
        chkpts = [
            pretrained_dir / f"0{split_idx}/epoch=019.weights.pth",
            pretrained_dir / f"0{split_idx}/epoch=019.aux.dat"
        ]
        # Perform ensembling by averaging probabilities
        # across checkpoint predictions
        outputs = run_once(
            new_split,
            num_epochs=1,
            on_gpu=True,
            save_dir=None,
            arch_kwargs=arch_kwargs,
            loader_kwargs=loader_kwargs,
            preproc_func=transform_dict,
            pretrained=chkpts,
            BayesGNN=BayesGNN,
            data_types=omic_keys,
            concept_weight=None,
            use_histopath=use_histopath
        )
        pred_logit, feature = list(zip(*outputs))
        risk_scores = np.exp(np.array(pred_logit).squeeze())
        if aggregation == "CBM":
            concept_logit = feature
        else:
            concept_logit = np.random.normal(loc=0, scale=1, size=(len(risk_scores), 30))
        concept_logit = np.array(concept_logit).squeeze()

        # saving average features
        if save_features:
            risk_scores = risk_scores[len(tr_samples):]
            concept_logit = concept_logit[len(tr_samples):]
            graph_paths = [d["pathomics"] for d in new_split["infer"]]
            save_dir = pathlib.Path(graph_paths[0]).parents[0] / f"CL_{aggregation}" / f"0{split_idx}"
            mkdir(save_dir)
            for i, path in enumerate(graph_paths): 
                save_name = pathlib.Path(path).name.replace(".json", ".npy") 
                save_path = f"{save_dir}/{save_name}"
                np.save(save_path, feature[i])

        model = ConceptGraphArch(**arch_kwargs)
        model.load(*chkpts)
        if len(model.aux_model) == 0:
            concept_prob = 1 / (1 + np.exp(-concept_logit))
        else:
            concept_prob = []
            for i in range(concept_logit.shape[1]):
                scaler = model.aux_model[f"scaler{i+1}"]
                logit = concept_logit[:, i:i+1]
                prob = scaler.predict_proba(logit)[:, 1:2]
                concept_prob.append(prob)
            concept_prob = np.concatenate(concept_prob, axis=1)
        cum_prob.append(concept_prob)
        concept_label = (concept_prob > 0.5).astype(np.int8)
        cum_label.append(concept_label)
        concept_true = np.array([v[1]["concept"] for v in split["test"]])
        concept_true = np.array(concept_true).squeeze().astype(np.int8)
        cum_true.append(concept_true)

        # * Calculate split statistics
        tr_y = np.array([v[1]["label"] for v in tr_samples])
        tr_y = pd.DataFrame({'event': tr_y[:, 1].astype(bool), 'duration': tr_y[:, 0]})
        tr_y = tr_y.to_records(index=False)

        te_y = np.array([v[1]["label"] for v in split["test"]])
        te_y = pd.DataFrame({'event': te_y[:, 1].astype(bool), 'duration': te_y[:, 0]})
        te_y = te_y.to_records(index=False)
        C_index = concordance_index_censored(te_y["event"], te_y["duration"], risk_scores)[0]
        C_index_ipcw = concordance_index_ipcw(tr_y, te_y, risk_scores)[0]
        lower, upper = np.percentile(te_y["duration"], [10, 90])
        times = np.arange(lower, upper + 1, 7)
        _, mean_auc = cumulative_dynamic_auc(tr_y, te_y, risk_scores, times)

        cum_stats.append(
            {
                "C-Index": np.array(C_index),
                "C-IPCW": np.array(C_index_ipcw),
                "C-AUC": np.array(mean_auc)
            }
        )
    cindex_list = [stat["C-Index"] for stat in cum_stats]
    cipcw_list = [stat["C-IPCW"] for stat in cum_stats]
    cauc_list = [stat["C-AUC"] for stat in cum_stats]
    avg_stat = {
        "C-Index mean": np.stack(cindex_list, axis=0).mean(axis=0),
        "C-Index std": np.stack(cindex_list, axis=0).std(axis=0),
        "C-IPCW mean": np.stack(cipcw_list, axis=0).mean(axis=0),
        "C-IPCW std": np.stack(cipcw_list, axis=0).std(axis=0),
        "C-AUC mean": np.stack(cauc_list, axis=0).mean(axis=0),
        "C-AUC std": np.stack(cauc_list, axis=0).std(axis=0),
    }
    prob = np.concatenate(cum_prob, axis=0)
    label = np.concatenate(cum_label, axis=0)
    true = np.concatenate(cum_true, axis=0)
    acc_list, f1_list, auc_list, ap_list = [], [], [], []
    for i in range(concept_true.shape[1]):
        acc_list.append(acc_scorer(true[:, i], label[:, i]))
        f1_list.append(f1_scorer(true[:, i], label[:, i]))
        auc_list.append(auroc_scorer(true[:, i], prob[:, i]))
        ap_list.append(ap_scorer(true[:, i], prob[:, i]))
    avg_stat.update({
        "ACC": np.array(acc_list),
        "F1": np.array(f1_list),
        "AUC": np.array(auc_list),
        "AP": np.array(ap_list)
    })
    for k, v in avg_stat.items(): 
        print(k, v.mean())
        top10 = np.sort(v, axis=None)[-10:]
        print(f"{k} top10", top10.mean())

    # save results
    save_path = pretrained_dir / f"concept_classification_results.json"
    with save_path.open("w") as handle:
        new_dict = {k: v.tolist() for k, v in avg_stat.items()}
        json.dump(new_dict, handle)
    return avg_stat

def test(
    graph_path,
    scaler_path,
    num_node_features,
    num_concepts,
    pretrained_model,
    conv="MLP",
    dropout=0,
    omic_keys=["pathomics"],
    aggregation="CBM",
    use_histopath=False
):
    """node classification 
    """
    node_scalers = [joblib.load(scaler_path[k]) for k in omic_keys] 
    transform_dict = {k: s.transform for k, s in zip(omic_keys, node_scalers)}
    
    loader_kwargs = {
        "num_workers": 1,
        "batch_size": 1,
    }
    if use_histopath: 
        num_node_features["pathomics"] = num_node_features["pathomics"] + 35
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_concept": num_concepts,
        "dim_target": 1,
        "layers": [384, 256, 128, 64], # [16, 16, 8]
        "dropout": dropout,  #0.5
        "conv": conv,
        "keys": omic_keys,
        "aggregation": aggregation
    }

    # BayesGNN = True
    new_split = {"infer": [graph_path]}
    outputs = run_once(
        new_split,
        num_epochs=1,
        save_dir=None,
        on_gpu=False,
        pretrained=pretrained_model,
        arch_kwargs=arch_kwargs,
        loader_kwargs=loader_kwargs,
        preproc_func=transform_dict,
        data_types=omic_keys,
        use_histopath=use_histopath
    )

    pred_logit, concept_logit, attention = list(zip(*outputs))
    hazard = np.exp(np.array(pred_logit).squeeze())
    concept_logit = np.array(concept_logit).squeeze()
    concept_prob = 1 / (1 + np.exp(-concept_logit))
    concept_label = (concept_prob > 0.5).astype(np.int8)
    attention = np.array(attention).squeeze()
    return hazard, concept_label, attention

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', default="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/TCGA/WSI")
    parser.add_argument('--dataset', default="TCGA-RCC", type=str)
    parser.add_argument('--modality', default="CT", type=str)
    parser.add_argument('--save_pathomics_dir', default="/home/sg2162/rds/hpc-work/Experiments/pathomics", type=str)
    parser.add_argument('--save_radiomics_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--save_clinical_dir', default="/home/sg2162/rds/hpc-work/Experiments/clinical", type=str)
    parser.add_argument('--mode', default="wsi", choices=["tile", "wsi"], type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--pathomics_mode', default="conch", choices=["cnn", "vit", "uni", "conch", "chief"], type=str)
    parser.add_argument('--pathomics_dim', default=512, choices=[2048, 384, 1024, 512, 768], type=int)
    parser.add_argument('--pathomics_aggregated_mode', default="CBM", choices=["None", "ABMIL", "CBM"], type=str, 
                        help="if graph has been aggregated, specify which mode, defaut is none"
                        )
    parser.add_argument('--num_concepts', default=39, type=int)
    parser.add_argument('--survival', default="OS", choices=["OS", "DFS"], type=str)
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
    save_pathomics_dir = pathlib.Path(f"{args.save_pathomics_dir}/{args.dataset}_{args.mode}_pathomic_features/{args.pathomics_mode}")
    save_clinical_dir = pathlib.Path(f"{args.save_clinical_dir}")
    save_model_dir = pathlib.Path(f"{args.save_pathomics_dir}/{args.dataset}_{args.mode}_models/{args.pathomics_mode}")

    # request survial data by GDC API
    # project_ids = ["TCGA-KIRP", "TCGA-KIRC", "TCGA-KICH"]
    # project_ids = ["TCGA-KIRC"]
    # request_survival_data(project_ids, save_clinical_dir)

    # plot survival curve
    # plot_survival_curve(save_clinical_dir)

    # survival analysis
    pathomics_paths = [save_pathomics_dir / f"{p.stem}.json" for p in wsi_paths]
    # pathomics_paths = sorted(pathlib.Path(save_pathomics_dir).rglob("*.json"))
    df_concepts, matched_i = matched_concepts_graph(save_clinical_dir, pathomics_paths)
    matched_pathomics_paths = [pathomics_paths[i] for i in matched_i]
    # df_concepts = df_concepts[["Tumor type: clear cell", "Tumor type: papillary", "Tumor type: chromophobe"]]
    # df_concepts = df_concepts.drop(columns=["Tumor type: papillary", "Tumor type: chromophobe"])
    concepts = df_concepts.to_numpy(dtype=np.float32)
    selected = np.array(concepts).sum(axis=0) > 50
    concepts = concepts[:, selected].tolist()
    concept_weight = len(concepts) / np.array(concepts).sum(axis=0)
    args.num_concepts = len(concept_weight)
    print("The number of selected concepts:", args.num_concepts)
    concept_names = list(df_concepts.columns)
    concept_names = [concept_names[i] for i, v in enumerate(selected.tolist()) if v]
    # for name in concept_names: print(name)

    # split data set
    num_folds = 5
    test_ratio = 0.2
    train_ratio = 0.8 * 0.9
    valid_ratio = 0.8 * 0.1
    data_types = ["pathomics"]
    # stages=["Stage I", "Stage II"]

    if args.survival == "OS":
        df_clinical, matched_i = matched_survival_graph(save_clinical_dir, matched_pathomics_paths)
    # TCGA-KIRC, Disease Free Survival, M0 patients
    elif args.survival == "DFS":
        df_clinical, matched_i = matched_survival_graph(
            save_clinical_dir, matched_pathomics_paths, 
            dataset="TCGA-KIRC", survival="DFS", metastasis=["M0"])
    labels = df_clinical[['duration', 'event']].to_numpy(dtype=np.float32).tolist()
    concepts = [concepts[i] for i in matched_i]
    print("The number of samples for each class:", np.sum(np.array(concepts) == 1.0, axis=0))
    demographics = df_clinical[['gender', 'race']].values.tolist()
    y = [{"concept": c, "label": l, "demographic": d} for c, l, d in zip(concepts, labels, demographics)]
    matched_pathomics_paths = [matched_pathomics_paths[i] for i in matched_i]
    kp = data_types[0]
    matched_graph_paths = [{kp : p} for p in matched_pathomics_paths]
    splits = generate_data_split(
        x=matched_graph_paths,
        y=y,
        train=train_ratio,
        valid=valid_ratio,
        test=test_ratio,
        num_folds=num_folds
    )
    mkdir(save_model_dir)
    split_path = f"{save_model_dir}/concept_pathomics_{args.pathomics_mode}_splits.dat"
    joblib.dump(splits, split_path)
    splits = joblib.load(split_path)
    num_train = len(splits[0]["train"])
    logging.info(f"Number of training samples: {num_train}.")
    num_valid = len(splits[0]["valid"])
    logging.info(f"Number of validating samples: {num_valid}.")
    num_test = len(splits[0]["test"])
    logging.info(f"Number of testing samples: {num_test}.")

    # survival analysis
    survival(
        split_path=split_path,
        concepts=concept_names,
        pathomics_keys=None,
        pathomics_aggregation=False,
        pathomics_aggregated_mode=args.pathomics_aggregated_mode,
        used=["concepts", "pathomics"][1], 
        n_jobs=8,
        model=["RSF", "CoxPH", "Coxnet", "FastSVM"][1],
        scorer=["cindex", "cindex-ipcw", "auc", "ibs"][0],
        feature_selection=False,
        n_bootstraps=0
    )

    # compute mean and std on training data for normalization 
    # splits = joblib.load(split_path)
    # train_graph_paths = [path for path, _ in splits[0]["train"]]
    # loader = ConceptGraphDataset(train_graph_paths, mode="infer", data_types=data_types)
    # loader = DataLoader(
    #     loader,
    #     num_workers=8,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    # )
    # omic_features = [{k: v.x.numpy() for k in data_types} for v in loader]
    # omics_modes = {"pathomics": args.pathomics_mode}
    # for k, v in omics_modes.items():
    #     node_features = [d[k] for d in omic_features]
    #     node_features = np.concatenate(node_features, axis=0)
    #     node_scaler = StandardScaler(copy=False)
    #     node_scaler.fit(node_features)
    #     scaler_path = f"{save_model_dir}/concept_{k}_{v}_scaler.dat"
    #     joblib.dump(node_scaler, scaler_path)

    # training
    omics_modes = {"pathomics": args.pathomics_mode}
    omics_dims = {"pathomics": args.pathomics_dim}
    split_path = f"{save_model_dir}/concept_pathomics_{args.pathomics_mode}_splits.dat"
    scaler_paths = {k: f"{save_model_dir}/concept_{k}_{v}_scaler.dat" for k, v in omics_modes.items()}
    # training(
    #     num_epochs=args.epochs,
    #     split_path=split_path,
    #     scaler_path=scaler_paths,
    #     num_node_features=omics_dims,
    #     num_concepts=args.num_concepts,
    #     model_dir=save_model_dir,
    #     conv="GATConv",
    #     n_works=8,
    #     batch_size=32,
    #     dropout=0.5,
    #     BayesGNN=False,
    #     omic_keys=list(omics_modes.keys()),
    #     aggregation=["ABMIL", "CBM"][1],
    #     concept_weight=None,
    #     use_histopath=False
    # )

    # visualize concept attention
    # splits = joblib.load(split_path)
    # graph_path = splits[0]["test"][15][0] #14
    # graph_name = pathlib.Path(graph_path["pathomics"]).stem
    # print("Visualizing on wsi:", graph_name)
    # wsi_path = [p for p in wsi_paths if p.stem == graph_name][0]
    # pretrained_dir = save_model_dir / "30CL_unweighted_calibrated_Survival_Prediction_GATConv_CBM_/00"
    # chkpts = [
    #     pretrained_dir / "epoch=019.weights.pth",
    #     pretrained_dir / "epoch=019.aux.dat"
    # ]
    # hazard, concept_label, attention = test(
    #     graph_path=graph_path,
    #     scaler_path=scaler_paths,
    #     num_node_features=omics_dims,
    #     num_concepts=args.num_concepts,
    #     pretrained_model=chkpts,
    #     conv="GATConv",
    #     dropout=0.5,
    #     omic_keys=list(omics_modes.keys()),
    #     aggregation=["ABMIL", "CBM"][1],
    #     use_histopath=False
    # )
    # save_wsi = True
    # for i, concept in enumerate(concept_names):
    #     if concept_label[i] == 1:
    #         visualize_pathomic_graph(
    #             wsi_path=wsi_path,
    #             graph_path=graph_path["pathomics"],
    #             label=attention[:, i],
    #             show_map=True,
    #             save_name=f"concept{i}",
    #             save_title=concept,
    #             resolution=args.resolution,
    #             units=args.units,
    #             save_wsi=save_wsi
    #         )
    #         save_wsi = False

    # visualize conch prediction
    # graph_path = graph_path["pathomics"]
    # label_path = f"{graph_path}".replace(".json", ".label.npy")
    # visualize_pathomic_graph(
    #     wsi_path=wsi_path,
    #     graph_path=graph_path,
    #     magnify=False,
    #     # label=label_path,
    #     # show_map=True,
    #     # save_name=f"conch",
    #     # save_title="CONCH classification",
    #     resolution=args.resolution,
    #     units=args.units
    # )

    # inference and visualize metrics
    # outputs = inference(
    #     split_path=split_path,
    #     scaler_path=scaler_paths,
    #     num_node_features=omics_dims,
    #     num_concepts=args.num_concepts,
    #     pretrained_dir=save_model_dir,
    #     conv="GATConv",
    #     n_works=32,
    #     batch_size=32,
    #     dropout=0.0,
    #     BayesGNN=False,
    #     omic_keys=list(omics_modes.keys()),
    #     aggregation=["ABMIL", "CBM"][1],
    #     use_histopath=False,
    #     save_features=True
    # )
    # counts = np.sum(np.array(concepts) == 1.0, axis=0)
    # acc = outputs["ACC"]
    # sorted_index = np.argsort(acc).tolist()
    # sorted_names = [concept_names[i] for i in sorted_index]
    # sorted_counts = counts[sorted_index] / len(concepts)
    # sorted_acc = acc[sorted_index]
    # x = np.arange(len(concept_names)) + 1
    # plt.figure(figsize=(20, 15))
    # ax = plt.subplot(1,1,1)
    # ax.plot(x, sorted_counts, color='green', marker='o', label="ratio of concept")
    # ax.plot(x, sorted_acc, color='blue', marker='^', label="ACC")
    # ax.set_xticks(x)
    # ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    # ax.legend()
    # plt.subplots_adjust(bottom=0.3)
    # plt.savefig("a_07explainable_AI/concept_acc_auroc.jpg")

    # subtyping
    # pretrained_dir = save_model_dir / f"30CL_unweighted_calibrated_Survival_Prediction_GATConv_CBM_"
    # json_path = pretrained_dir / f"concept_classification_results.json"
    # outputs = load_json(json_path)
    # acc = np.array(outputs["ACC"])
    # sorted_index = np.argsort(acc).tolist()
    # sorted_names = [concept_names[i] for i in sorted_index]
    # sorted_acc = acc[sorted_index].tolist()
    # print("ACC", {k: v for k, v in zip(sorted_names[-3:], sorted_acc[-3:])})
    # f1 = np.array(outputs["F1"])
    # sorted_f1 = f1[sorted_index].tolist()
    # print("F1", {k: v for k, v in zip(sorted_names[-3:], sorted_f1[-3:])})
    # auc = np.array(outputs["AUC"])
    # sorted_auc = auc[sorted_index].tolist()
    # print("AUC", {k: v for k, v in zip(sorted_names[-3:], sorted_auc[-3:])})
    # ap = np.array(outputs["AP"])
    # sorted_ap = ap[sorted_index].tolist()
    # print("AP", {k: v for k, v in zip(sorted_names[-3:], sorted_ap[-3:])})

    # plot radar chart
    # import plotly.graph_objects as go
    # metric = "AUC"
    # save_model_dir = pathlib.Path(f"{args.save_pathomics_dir}/{args.dataset}_{args.mode}_models")
    # pretrained_dir = save_model_dir / "uni" / f"28CL_unweighted_calibrated_DFS_Prediction_GATConv_CBM_"
    # json_path = pretrained_dir / f"concept_classification_results.json"
    # outputs = load_json(json_path)
    # acc = np.array(outputs[metric])
    # sorted_index = np.argsort(acc).tolist()
    # sorted_names = [concept_names[i] for i in sorted_index]
    # uni_sorted_acc = acc[sorted_index].tolist()

    # pretrained_dir = save_model_dir / "vit" / f"28CL_unweighted_calibrated_DFS_Prediction_GATConv_CBM_"
    # json_path = pretrained_dir / f"concept_classification_results.json"
    # outputs = load_json(json_path)
    # acc = np.array(outputs[metric])
    # vit_sorted_acc = acc[sorted_index].tolist()

    # pretrained_dir = save_model_dir / "conch" / f"28CL_unweighted_calibrated_DFS_Prediction_GATConv_CBM_"
    # json_path = pretrained_dir / f"concept_classification_results.json"
    # outputs = load_json(json_path)
    # acc = np.array(outputs[metric])
    # conch_sorted_acc = acc[sorted_index].tolist()

    # pretrained_dir = save_model_dir / "chief" / f"28CL_unweighted_calibrated_DFS_Prediction_GATConv_CBM_"
    # json_path = pretrained_dir / f"concept_classification_results.json"
    # outputs = load_json(json_path)
    # acc = np.array(outputs[metric])
    # chief_sorted_acc = acc[sorted_index].tolist()

    # fig = go.Figure()

    # fig.add_trace(go.Scatter(
    #     x=sorted_names, 
    #     y=vit_sorted_acc,
    #     mode='lines+markers',
    #     name='HIPT+CBM'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=sorted_names, 
    #     y=uni_sorted_acc,
    #     mode='lines+markers',
    #     name='UNI+CBM'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=sorted_names, 
    #     y=conch_sorted_acc,
    #     mode='lines+markers',
    #     name='CONCH+CBM'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=sorted_names, 
    #     y=chief_sorted_acc,
    #     mode='lines+markers',
    #     name='CHIEF+CBM'
    # ))

    # fig.update_layout(
    #         xaxis=dict(
    #             title=dict(
    #                 text='Kidney Pathological Concept'
    #             ),
    #             tickangle=30,
    #             tickfont=dict(
    #             size=12
    #             ),
    #         ),
    #         yaxis=dict(
    #             title=dict(
    #                 text='AUC'
    #             )
    #         ),
    #         margin=dict(t=10, l=50, r=30),
    #         legend=dict(
    #             x=0.1, y=0.9,
    #             xanchor='left', yanchor='top'
    #         )
    # )

    # fig.write_image(f"a_07explainable_AI/concept_learning_metric.png", width=1000, height=500, scale=2)


    