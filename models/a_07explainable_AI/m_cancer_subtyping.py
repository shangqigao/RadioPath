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
from collections import Counter
from scipy.stats import zscore
from torch_geometric.loader import DataLoader
from tiatoolbox import logger
from tiatoolbox.utils.misc import save_as_json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import balanced_accuracy_score as acc_scorer

from common.m_utils import mkdir, select_wsi, load_json, create_pbar, rm_n_mkdir, reset_logging, recur_find_ext, select_checkpoints

from models.a_05feature_aggregation.m_gnn_cancer_subtyping import SubtypingGraphDataset, SubtypingGraphArch, SubtypingBayesGraphArch
from models.a_05feature_aggregation.m_gnn_survival_analysis import ScalarMovingAverage



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

def matched_survival_graph(save_clinical_dir, save_graph_paths):
    df = pd.read_csv(f"{save_clinical_dir}/TCGA_PanKidney_survival_data.csv")
    print("Survival data strcuture:", df.shape)

    # filter graph properties 
    ids = df['submitter_id']
    names = [p.stem for p in save_graph_paths]
    matched_index, matched_i = [], []
    for index, id in ids.items():
        for i, name in enumerate(names):
            if id in name:
                matched_index.append(index)
                matched_i.append(i)
    df = df.loc[matched_index]
    return df, matched_i

def BalancedShuffleSplitter(y, sample_size, random_seed=None):
    uniq_levels = np.unique(y)

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for level in uniq_levels:
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    train_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        train_idx += over_sample_idx
    np.random.shuffle(train_idx)

    if  ((len(train_idx)) == (sample_size*len(uniq_levels))):
        print('number of sampled example:', sample_size*len(uniq_levels), 'number of sample per class:', sample_size, ' #classes:', len(list(set(uniq_levels))))
    else:
        print('number of samples is wrong ')

    labels_train = y[train_idx]
    labels, values = zip(*Counter(labels_train).items())
    print('number of classes ', len(list(set(labels_train))))
    check = all(x == values[0] for x in values)
    if check == True:
        print('Good! all classes are balanced')
    else:
        print('Repeat again! classes are imbalanced')
    
    test_idx = list(set([i for i in range(len(y))]) - set(train_idx))
    return train_idx, test_idx

def generate_data_split(
        x: list,
        y: list,
        train: float,
        valid: float,
        test: float,
        num_folds: int,
        seed: int = 5,
        balanced=False
):
    """Helper to generate splits
    Args:
        x (list): a list of image paths
        y (list): a list of annotation paths
        train (float or int): if int, number of training samples per class
        if float, ratio of training samples
        test (float): ratio of testing samples
        num_folds (int): number of folds for cross-validation
        seed (int): random seed
        balanced (bool): if true, sampling equal number of samples for each class
    Returns:
        splits (list): a list of folds, each fold consists of train, valid, and test splits
    """
    if balanced:
        assert train >= 1, "The number training samples should be no less than 1"
        assert test < 1, "The ratio of testing samples should be less than 1"
        outer_splitter = StratifiedShuffleSplit(
            n_splits=num_folds,
            test_size=test,
            random_state=seed,
        )
    else:
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

    l = np.array(y)

    splits = []
    random_seed = seed
    for train_valid_idx, test_idx in outer_splitter.split(x, l):
        test_x = [x[idx] for idx in test_idx]
        test_y = [y[idx] for idx in test_idx]
        x_ = [x[idx] for idx in train_valid_idx]
        y_ = [y[idx] for idx in train_valid_idx]
        l_ = [l[idx] for idx in train_valid_idx]

        l_ = np.array(l_)
        if balanced:
            train_idx, valid_idx = BalancedShuffleSplitter(l_, train, random_seed)
            random_seed += 1
        else:
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
        model = SubtypingBayesGraphArch(**arch_kwargs)
        kl = {"loss": bnn.BKLLoss(), "weight": 0.1}
    else:
        model = SubtypingGraphArch(**arch_kwargs)
        kl = None
    if pretrained is not None:
        model.load(*pretrained)
    model = model.to("cuda")
    loss = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, **optim_kwargs)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        ds = SubtypingGraphDataset(
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

                if "train" in loader_name:
                    scaler = PlattScaling(solver="saga", multi_class="multinomial")
                    scaler.fit(logit, true)
                    model.aux_model["scaler"] = scaler
                scaler = model.aux_model["scaler"]
                prob = scaler.predict_proba(logit)

                label = np.argmax(prob, axis=1)
                val = acc_scorer(label, true)
                logging_dict[f"{loader_name}-acc"] = val

                val = auroc_scorer(true, prob, multi_class="ovr")
                logging_dict[f"{loader_name}-auroc"] = val

                onehot = np.eye(logit.shape[1])[true]
                val = auprc_scorer(onehot, prob)
                logging_dict[f"{loader_name}-auprc"] = val

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
        model_dir,
        conv="GCNConv",
        n_works=32,
        batch_size=32,
        dropout=0.,
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
        "dim_target": 3,
        "layers": [512, 384], # [16, 16, 8]
        "dropout": dropout,  #0.5
        "conv": conv,
    }
    if BayesGNN:
        model_dir = model_dir / f"Bayes_Cancer_Subtyping_{conv}"
    else:
        model_dir = model_dir / f"Cancer_Subtyping_{conv}"
    optim_kwargs = {
        "lr": 1e-4,
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
        batch_size=32
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
            # * re-calibrate logit to probabilities
            model = SubtypingGraphArch(**arch_kwargs)
            model.load(*chkpt_info)
            scaler = model.aux_model["scaler"]
            chkpt_results = list(zip(*chkpt_results))
            chkpt_results = np.array(chkpt_results).squeeze()
            chkpt_results = scaler.predict_proba(chkpt_results)
            cum_results.append(chkpt_results)
        cum_results = np.array(cum_results)
        cum_results = np.squeeze(cum_results)

        prob = cum_results
        if len(cum_results.shape) == 3:
            prob = np.mean(cum_results, axis=0)

        # * Calculate split statistics
        true = np.array([v[1] for v in split["test"]])
        onehot = np.eye(prob.shape[1])[true] 
        
        ## compute per-class accuracy
        pred = np.argmax(prob, axis=1)
        uids = np.unique(true)
        acc_scores = []
        for i in range(len(uids)):
            indices = true == uids[i]
            score = acc_scorer(true[indices], pred[indices])
            acc_scores.append(score)

        cum_stats.append(
            {
                "acc": np.array(acc_scores),
                "auroc": auroc_scorer(true, prob, average=None, multi_class="ovr"), 
                "auprc": auprc_scorer(onehot, prob, average=None),
            }
        )
        print(f"Fold-{split_idx}:", cum_stats[-1])
    acc_list = [stat["acc"] for stat in cum_stats]
    auroc_list = [stat["auroc"] for stat in cum_stats]
    auprc_list = [stat["auprc"] for stat in cum_stats]
    avg_stat = {
        "acc-mean": np.stack(acc_list, axis=0).mean(axis=0),
        "acc-std": np.stack(acc_list, axis=0).std(axis=0),
        "auroc-mean": np.stack(auroc_list, axis=0).mean(axis=0),
        "auroc-std": np.stack(auroc_list, axis=0).std(axis=0),
        "auprc-mean": np.stack(auprc_list, axis=0).mean(axis=0),
        "auprc-std": np.stack(auprc_list, axis=0).std(axis=0)
    }
    print(f"Avg:", avg_stat)
    return cum_stats

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', default="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/TCGA/WSI")
    parser.add_argument('--dataset', default="TCGA-RCC", type=str)
    parser.add_argument('--save_pathomics_dir', default="/home/sg2162/rds/hpc-work/Experiments/pathomics", type=str)
    parser.add_argument('--save_clinical_dir', default="/home/sg2162/rds/hpc-work/Experiments/clinical", type=str)
    parser.add_argument('--mode', default="wsi", choices=["tile", "wsi"], type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--feature_mode', default="uni", choices=["cnn", "vit", "uni", "conch"], type=str)
    parser.add_argument('--node_features', default=1024, choices=[2048, 384, 1024, 35], type=int)
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
    save_model_dir = pathlib.Path(f"{args.save_pathomics_dir}/{args.dataset}_{args.mode}_models/{args.feature_mode}")

    # request survial data by GDC API
    # project_ids = ["TCGA-KIRP", "TCGA-KIRC", "TCGA-KICH"]
    # request_survival_data(project_ids, save_clinical_dir)

    # plot survival curve
    # plot_survival_curve(save_clinical_dir)

    # match and split data set
    graph_paths = [save_pathomics_dir / f"{p.stem}.MST.json" for p in wsi_paths]
    df, matched_i = matched_survival_graph(save_clinical_dir, graph_paths)
    lab_dict = {"TCGA-KIRC": 0, "TCGA-KIRP": 1, "TCGA-KICH": 2}
    y = df["project_id"].apply(lambda x: lab_dict[x]).to_numpy(dtype=np.int32).tolist()
    matched_graph_paths = [graph_paths[i] for i in matched_i]

    num_folds = 5
    balanced_sampling = True
    if balanced_sampling:
        test_ratio = 0.5
        train = 2
        valid_ratio = 1 - (train / len(matched_graph_paths)) - test_ratio
    else:
        test_ratio = 0.2
        train = 0.8 * 0.9
        valid_ratio = 0.8 * 0.1
    splits = generate_data_split(
        x=matched_graph_paths,
        y=y,
        train=train,
        valid=valid_ratio,
        test=test_ratio,
        num_folds=num_folds,
        balanced=balanced_sampling
    )
    mkdir(save_model_dir)
    split_path = f"{save_model_dir}/subtyping_splits.dat"
    joblib.dump(splits, split_path)
    splits = joblib.load(split_path)
    num_train = len(splits[0]["train"])
    logging.info(f"Number of training samples: {num_train}.")
    num_valid = len(splits[0]["valid"])
    logging.info(f"Number of validating samples: {num_valid}.")
    num_test = len(splits[0]["test"])
    logging.info(f"Number of testing samples: {num_test}.")

    # compute mean and std on training data for normalization 
    splits = joblib.load(split_path)
    train_graph_paths = [path for path, _ in splits[0]["train"]]
    loader = SubtypingGraphDataset(train_graph_paths, mode="infer")
    loader = DataLoader(
        loader,
        num_workers=8,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    node_features = [v.x.numpy() for v in loader]
    node_features = np.concatenate(node_features, axis=0)
    node_scaler = StandardScaler(copy=False)
    node_scaler.fit(node_features)
    scaler_path = f"{save_model_dir}/subtyping_node_scaler.dat"
    joblib.dump(node_scaler, scaler_path)

    # training
    training(
        num_epochs=args.epochs,
        split_path=split_path,
        scaler_path=scaler_path,
        num_node_features=args.node_features,
        model_dir=save_model_dir,
        conv="MLP",
        n_works=8,
        batch_size=1,
        dropout=0.1,
        BayesGNN=True
    )