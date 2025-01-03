import sys
sys.path.append('../')

import requests
import argparse
import pathlib
import logging
import warnings
import joblib
import copy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torchbnn as bnn

from scipy.stats import zscore
from torch_geometric.loader import DataLoader
from tiatoolbox import logger
from tiatoolbox.utils.misc import save_as_json

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored

from sklearn import set_config
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression as PlattScaling

from common.m_utils import mkdir, select_wsi, load_json, create_pbar, rm_n_mkdir, reset_logging, recur_find_ext, select_checkpoints

from models.a_05feature_aggregation.m_gnn_gene_mutation import MutationGraphDataset, MutationGraphArch, MutationBayesGraphArch
from models.a_05feature_aggregation.m_gnn_gene_mutation import ScalarMovingAverage



def request_survival_data(project_ids, save_dir):
    fields = [
        "case_id",
        "submitter_id",
        "project.project_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.ajcc_pathologic_stage"
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
            'ajcc_pathologic_stage': case['diagnoses'][0].get('ajcc_pathologic_stage', None),
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

def prepare_radiomic_properties(data_dict, prop_keys):
    properties = {}
    for key, value in data_dict.items():
        selected = [((k in key) and ("diagnostics" not in key)) for k in prop_keys]
        if any(selected): properties[key] = value
    return properties

def prepare_graph_features(
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
    label_path = f"{graph_path}".replace(".MST.json", ".label.npy")
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
        kmeans = KMeans(n_clusters=1)
        feat_list = kmeans.fit(feature).cluster_centers_
        feat_list = feat_list.squeeze().tolist()
        
    feat_dict = {}
    for i, feat in enumerate(feat_list):
        k = f"graph.feature{i}"
        feat_dict[k] = feat
    return {f"{idx}": feat_dict}

def load_wsi_level_features(idx, wsi_feature_path):
    feat_list = np.array(np.load(wsi_feature_path)).tolist()
    feat_dict = {}
    for i, feat in enumerate(feat_list):
        k = f"graph.feature{i}"
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

def matched_survival_graph(save_clinical_dir, save_graph_paths, dataset="TCGA-RCC", stages=None):
    df = pd.read_csv(f"{save_clinical_dir}/{dataset}_survival_data.csv")
    
    # Prepare the survival data
    df['event'] = df['vital_status'].apply(lambda x: True if x == 'Dead' else False)
    df['duration'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
    df = df[df['duration'].notna()]
    if stages is not None:
        df = df[df['ajcc_pathologic_stage'].isin(stages)]
    print("Survival data strcuture:", df.shape)

    # filter graph properties 
    graph_names = [pathlib.Path(p).stem for p in save_graph_paths]
    graph_ids = [f"{n}".split("-")[0:3] for n in graph_names]
    graph_ids = ["-".join(d) for d in graph_ids]
    df = df[df["submitter_id"].isin(graph_ids)]
    matched_indices = [graph_ids.index(d) for d in df["submitter_id"]]
    return df, matched_indices

def matched_pathomics_radiomics(save_pathomics_paths, save_radiomics_paths, save_clinical_dir, dataset="TCGA-RCC"):
    df = pd.read_csv(f"{save_clinical_dir}/TCIA_{dataset}_mappings.csv")

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
        radiomics_list = [load_json(p) for p in matched_radiomics_paths]
        radiomics_list = [prepare_radiomic_properties(p, radiomics_keys) for p in radiomics_list]
        df_radiomics = pd.DataFrame(radiomics_list)
        print("Selected radiomic properties:", df_radiomics.shape)
        print(df_radiomics.head())


    # Prepare WSI-level features
    selected_graph_paths = [save_pathomics_paths[i] for i in matched_i]
    if aggregation:
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(prepare_graph_features)(idx, graph_path)
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

    l = np.array([status for _, status in y])

    splits = []
    for train_valid_idx, test_idx in outer_splitter.split(x, l):
        test_x = [x[idx] for idx in test_idx]
        test_y = [y[idx] for idx in test_idx]
        x_ = [x[idx] for idx in train_valid_idx]
        y_ = [y[idx] for idx in train_valid_idx]
        l_ = [l[idx] for idx in train_valid_idx]

        train_idx, valid_idx = next(iter(inner_splitter.split(x_, l_)))
        valid_x = [x_[idx] for idx in valid_idx]
        valid_y = [y_[idx] for idx in valid_idx]
        train_x = [x_[idx] for idx in train_idx]
        train_y = [y_[idx] for idx in train_idx]

        assert len(set(train_x).intersection(set(valid_x))) == 0
        assert len(set(valid_x).intersection(set(test_x))) == 0
        assert len(set(train_x).intersection(set(test_x))) == 0

        splits.append(
            {
                "train": list(zip(train_x, train_y)),
                "valid": list(zip(valid_x, valid_y)),
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
        BayesGNN=False
):
    """running the inference or training loop once"""
    if loader_kwargs is None:
        loader_kwargs = {}

    if arch_kwargs is None:
        arch_kwargs = {}

    if optim_kwargs is None:
        optim_kwargs = {}

    if BayesGNN:
        model = MutationBayesGraphArch(**arch_kwargs)
        kl = {"loss": bnn.BKLLoss(), "weight": 0.1}
    else:
        model = MutationGraphArch(**arch_kwargs)
        kl = None
    if pretrained is not None:
        model.load(*pretrained)
    model = model.to("cuda")
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, **optim_kwargs)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        ds = SurvivalGraphDataset(
            subset, 
            mode=subset_name, 
            preproc=preproc_func
        )
        loader_dict[subset_name] = DataLoader(
            ds,
            drop_last=subset_name == "train",
            shuffle=subset_name == "train",
            **_loader_kwargs,
        )

    for epoch in range(num_epochs):
        logger.info("EPOCH: %03d", epoch)
        for loader_name, loader in loader_dict.items():
            step_output = []
            ema = ScalarMovingAverage()
            pbar = create_pbar(loader_name, len(loader))
            for step, batch_data in enumerate(loader):
                if loader_name == "train":
                    output = model.train_batch(model, batch_data, on_gpu, loss, optimizer, kl)
                    ema({"loss": output[0]})
                    pbar.postfix[1]["step"] = step
                    pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
                else:
                    output = model.infer_batch(model, batch_data, on_gpu)
                    batch_size = loader_kwargs["batch_size"]
                    batch_size = output[0].shape[0]
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
                logit, true = output
                logit = np.array(logit).squeeze()
                true = np.array(true).squeeze()
                event_status = true[:, 1] > 0
                event_time = true[:, 0]
                cindex = concordance_index_censored(event_status, event_time, logit)[0]
                logging_dict[f"{loader_name}-Cindex"] = cindex

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
                )
        lr_scheduler.step()
    
    return step_output


def training(
        num_epochs,
        split_path,
        scaler_path,
        num_node_features,
        model_dir,
        conv="GCNConv",
        n_works=32,
        batch_size=32,
        dropout=0,
        BayesGNN=False
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
    node_scaler = joblib.load(scaler_path)
    
    loader_kwargs = {
        "num_workers": n_works, 
        "batch_size": batch_size,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 1,
        "layers": [512, 384], # [16, 16, 8]
        "dropout": dropout,  #0.5
        "conv": conv,
    }
    if BayesGNN:
        model_dir = model_dir / f"Bayes_Survival_Prediction_{conv}"
    else:
        model_dir = model_dir / f"Survival_Prediction_{conv}"
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
            preproc_func=node_scaler.transform,
            BayesGNN=BayesGNN
        )
    return

def inference(
        split_path,
        scaler_path,
        num_node_features,
        pretrained_dir,
        n_works=32,
        batch_size=32,
        BayesGNN=False
):
    """survival prediction
    """
    splits = joblib.load(split_path)
    node_scaler = joblib.load(scaler_path)
    
    loader_kwargs = {
        "num_workers": n_works,
        "batch_size": batch_size,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 1,
        "layers": [16, 16, 8],
        "dropout": 0.5,
        "conv": "GATConv"
    }
    pretrained_dir = pretrained_dir / f"Survival_Prediction_{arch_kwargs['conv']}"
    cum_stats = []
    for split_idx, split in enumerate(splits):
        new_split = {"infer": [v[0] for v in split["test"]]}

        stat_files = recur_find_ext(f"{pretrained_dir}/{split_idx:02d}/", [".json"])
        stat_files = [v for v in stat_files if ".old.json" not in v]
        assert len(stat_files) == 1
        chkpts, _ = select_checkpoints(
            stat_files[0],
            top_k=1,
            metric="infer-valid-A-auroc",
        )
        # Perform ensembling by averaging probabilities
        # across checkpoint predictions

        cum_results = []
        for i, chkpt_info in enumerate(chkpts):
            chkpt_results = run_once(
                new_split,
                num_epochs=1,
                save_dir=None,
                pretrained=chkpt_info,
                arch_kwargs=arch_kwargs,
                loader_kwargs=loader_kwargs,
                preproc_func=node_scaler.transform,
            )
            chkpt_results = list(zip(*chkpt_results))
            chkpt_results = np.array(chkpt_results).squeeze()
            cum_results.append(chkpt_results)
        cum_results = np.array(cum_results)
        cum_results = np.squeeze(cum_results)

        prob = cum_results
        if len(cum_results.shape) == 2:
            prob = np.mean(cum_results, axis=0)

        # * Calculate split statistics
        true = np.array([v[1] for v in split["test"]])
        event_status = true[:, 1] > 0
        event_time = true[:, 0]
        cindex = concordance_index_censored(event_status, event_time, prob)[0]

        cum_stats.append(
            {
                "C-Index": np.array(cindex)
            }
        )
        print(f"Fold-{split_idx}:", cum_stats[-1])
    cindex_list = [stat["C-Index"] for stat in cum_stats]
    avg_stat = {
        "C-Index mean": np.stack(cindex_list, axis=0).mean(axis=0),
        "C-Index std": np.stack(cindex_list, axis=0).std(axis=0),
    }
    print(f"Avg:", avg_stat)
    return cum_stats

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
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--pathomics_mode', default="conch", choices=["cnn", "vit", "uni", "conch", "chief"], type=str)
    parser.add_argument('--pathomics_dim', default=1024, choices=[2048, 384, 1024, 35, 768], type=int)
    parser.add_argument('--radiomics_mode', default="pyradiomics", choices=["pyradiomics"], type=str)
    parser.add_argument('--radiomics_dim', default=107, choices=[107], type=int)
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
    save_radiomics_dir = pathlib.Path(f"{args.save_radiomics_dir}/{args.dataset}_{args.modality}_radiomic_features/{args.radiomics_mode}")
    save_clinical_dir = pathlib.Path(f"{args.save_clinical_dir}")
    save_model_dir = pathlib.Path(f"{args.save_pathomics_dir}/{args.dataset}_{args.mode}_models/{args.pathomics_mode}")

    # request survial data by GDC API
    # project_ids = ["TCGA-KIRP", "TCGA-KIRC", "TCGA-KICH"]
    # request_survival_data(project_ids, save_clinical_dir)

    # plot survival curve
    # plot_survival_curve(save_clinical_dir)

    # # split data set
    # num_folds = 5
    # test_ratio = 0.2
    # train_ratio = 0.8 * 0.9
    # valid_ratio = 0.8 * 0.1
    # graph_paths = [save_pathomics_dir / f"{p.stem}.MST.json" for p in wsi_paths]
    # # stages=["Stage I", "Stage II"]
    # df, matched_i = matched_survival_graph(save_clinical_dir, graph_paths)
    # y = df[['duration', 'event']].to_numpy(dtype=np.float32).tolist()
    # matched_graph_paths = [graph_paths[i] for i in matched_i]
    # splits = generate_data_split(
    #     x=matched_graph_paths,
    #     y=y,
    #     train=train_ratio,
    #     valid=valid_ratio,
    #     test=test_ratio,
    #     num_folds=num_folds,
    # )
    # mkdir(save_model_dir)
    # split_path = f"{save_model_dir}/survival_splits.dat"
    # joblib.dump(splits, split_path)
    # splits = joblib.load(split_path)
    # num_train = len(splits[0]["train"])
    # logging.info(f"Number of training samples: {num_train}.")
    # num_valid = len(splits[0]["valid"])
    # logging.info(f"Number of validating samples: {num_valid}.")
    # num_test = len(splits[0]["test"])
    # logging.info(f"Number of testing samples: {num_test}.")

    # # compute mean and std on training data for normalization 
    # splits = joblib.load(split_path)
    # train_graph_paths = [path for path, _ in splits[0]["train"]]
    # loader = SurvivalGraphDataset(train_graph_paths, mode="infer")
    # loader = DataLoader(
    #     loader,
    #     num_workers=8,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    # )
    # node_features = [v.x.numpy() for v in loader]
    # node_features = np.concatenate(node_features, axis=0)
    # node_scaler = StandardScaler(copy=False)
    # node_scaler.fit(node_features)
    # scaler_path = f"{save_model_dir}/survival_node_scaler.dat"
    # joblib.dump(node_scaler, scaler_path)

    # # training
    # training(
    #     num_epochs=args.epochs,
    #     split_path=split_path,
    #     scaler_path=scaler_path,
    #     num_node_features=args.node_features,
    #     model_dir=save_model_dir,
    #     conv="MLP",
    #     n_works=8,
    #     batch_size=32,
    #     dropout=0.1,
    #     BayesGNN=True
    # )

    # survival analysis
    aggregation = True # false if load wsi-level features else true
    if aggregation:
        pathomics_paths = [save_pathomics_dir / f"{p.stem}.MST.json" for p in wsi_paths]
    else:
        pathomics_paths = [save_pathomics_dir / f"{p.stem}.WSI.features.npy" for p in wsi_paths]

    # only use pathomics
    matched_pathomics_paths = pathomics_paths
    matched_radiomics_paths = None
    stages = ["Stage I", "Stage II"]

    # use radiomics and pathomics
    # class_name = ["kidney_and_mass", "mass", "tumour"][1]
    # radiomics_paths = list(save_radiomics_dir.glob(f"*{class_name}.{args.radiomics_mode}.json"))
    # matched_pathomics_indices, matched_radiomics_indices = matched_pathomics_radiomics(
    #     save_pathomics_paths=pathomics_paths,
    #     save_radiomics_paths=radiomics_paths,
    #     save_clinical_dir=save_clinical_dir,
    #     dataset=args.dataset
    # )
    # matched_pathomics_paths = [pathomics_paths[i] for i in matched_pathomics_indices]
    # matched_radiomics_paths = [radiomics_paths[i] for i in matched_radiomics_indices]
    # stages = None

    pathomic_properties = [
        "num_nodes", 
        "num_edges", 
        # "num_components", 
        "degree", 
        # "closeness", 
        # "graph_diameter",
        "graph_assortativity",
        # "mean_neighbor_degree"
    ]
    radiomic_propereties = [
        "shape",
        # "firstorder",
        "glcm",
        "gldm",
        # "glrlm",
        # "glszm",
        "ngtdm",
    ]
    integrated = ["all", "pathomics", "radiomics", "deep_pathomics", "radiopathomics", "radioDeepPathomics", "pathoDeepPathomics"]
    cox_proportional_hazard_regression(
        save_clinical_dir=save_clinical_dir,
        save_pathomics_paths=matched_pathomics_paths,
        save_radiomics_paths=matched_radiomics_paths,
        pathomics_keys=pathomic_properties,
        radiomics_keys=radiomic_propereties,
        l1_ratio=0.9,
        stages=stages,
        used=integrated[3],
        n_jobs=32,
        aggregation=aggregation,
        dataset=args.dataset
    )