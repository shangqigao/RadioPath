import sys
sys.path.append('../')

import pathlib
import logging
import copy
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm

import torchbnn as bnn

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import accuracy_score as acc_scorer
from sklearn.cluster import KMeans
from torch_geometric.loader import DataLoader

from tiatoolbox.utils.misc import save_as_json
from tiatoolbox import logger

from common.m_utils import mkdir, create_pbar, load_json, rm_n_mkdir, recur_find_ext
from common.m_utils import reset_logging, select_checkpoints, select_wsi_annotated

from models.a_02tissue_masking.m_tissue_masking import generate_wsi_tissue_mask
from models.a_03patch_extraction.m_patch_extraction import generate_tile_from_wsi
from models.a_04feature_extraction.m_feature_extraction import extract_wsi_feature
from models.a_05feature_aggregation.m_graph_construction import construct_wsi_graph
from models.a_05feature_aggregation.m_graph_construction import generate_node_label
from models.a_05feature_aggregation.m_graph_construction import visualize_graph
from models.a_05feature_aggregation.m_graph_construction import graph_feature_visualization
from models.a_05feature_aggregation.m_graph_neural_network import SlideGraphArch
from models.a_05feature_aggregation.m_graph_neural_network import SlideGraphDataset
from models.a_05feature_aggregation.m_graph_neural_network import ScalarMovingAverage
from models.a_05feature_aggregation.m_graph_neural_network import SlideBayesGraphArch
from models.a_05feature_aggregation.m_graph_neural_network import update_loss

torch.multiprocessing.set_sharing_strategy("file_system")

import warnings
warnings.filterwarnings('ignore')

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

    l = []
    for path in y:
        label = np.array(np.load(pathlib.Path(path)))
        negative = label[label == 0]
        positive = label[label > 0]
        ratio = np.prod(negative.shape) / (np.prod(positive.shape) + 1e-10)
        if ratio < 1:
            l.append(0)
        else:
            l.append(1)
    l = np.array(l)

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
        BayesGNN=False,
        arch_kwargs=None,
        optim_kwargs=None,
        loss_type = "CE",
        logit_adjusted=False,
        positive_subgraph=False,
        positive_negative=None,
        contrastive_learning=False
):
    """running the inference or training loop once"""
    if loader_kwargs is None:
        loader_kwargs = {}

    if arch_kwargs is None:
        arch_kwargs = {}

    if optim_kwargs is None:
        optim_kwargs = {}

    if BayesGNN:
        model = SlideBayesGraphArch(**arch_kwargs)
        kl = {"loss": bnn.BKLLoss(), "weight": 0.1}
    else:
        model = SlideGraphArch(**arch_kwargs)
        kl = None
    if pretrained is not None:
        model.load(*pretrained)
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, **optim_kwargs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    ## Set loss and whether to extract subgraph only containing positive nodes
    loss = update_loss(mode=loss_type)
    if logit_adjusted:
        probs, tau = [0.65, 0.15, 0.05, 0.15], 1
    else:
        probs, tau = None, 1

    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        ds = SlideGraphDataset(
            subset, 
            mode=subset_name, 
            preproc=preproc_func, 
            subgraph=positive_subgraph, 
            PN=positive_negative
        )
        loader_dict[subset_name] = DataLoader(
            ds,
            drop_last=subset_name == "train",
            shuffle=subset_name == "train",
            **_loader_kwargs,
        )

    best_acc, best_auroc, best_prior = 0., 0., 0.
    if contrastive_learning:
        centroid = np.random.normal(size=(1, 8, 6))
        intra_feat_sum, intra_mem_num = 0, 0
    else:
        centroid = None
    for epoch in range(num_epochs):
        logger.info("EPOCH: %03d", epoch)
        for loader_name, loader in loader_dict.items():
            step_output = []
            batch_indices = [0]
            ema = ScalarMovingAverage()
            pbar = create_pbar(loader_name, len(loader))
            for step, batch_data in enumerate(loader):
                if loader_name == "train":
                    output = model.train_batch(model, batch_data, on_gpu, loss, kl, optimizer, probs, tau, centroid)
                    ema({"loss": output[0]})
                    pbar.postfix[1]["step"] = step
                    pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
                    if contrastive_learning:
                        intra_feat_sum += output[1]
                        intra_mem_num += output[2]
                        ema({"centroid-maxstd": centroid.squeeze().std(axis=1).max()})
                else:
                    output = model.infer_batch(model, batch_data, on_gpu, loss_type)
                    batch_size = output[0].shape[0]
                    batch_indices.append(batch_indices[-1] + batch_size)
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
                if logit.ndim == 1:
                    label = np.zeros_like(true)
                    sigmoid = 1 / (1 + np.exp(-logit))
                    label[sigmoid > 0.5] = 1
                    val = acc_scorer(label[true > 0], true[true > 0])
                else:
                    label = np.argmax(logit, axis=1)
                    val = acc_scorer(label, true)
                logging_dict[f"{loader_name}-acc"] = val

                logit = logit.reshape(-1, 1) if logit.ndim == 1 else logit
                if "train" in loader_name:
                    if logit.shape[1] == 1:
                        scaler = PlattScaling()
                    else:
                        scaler = PlattScaling(solver="saga", multi_class="multinomial")
                    scaler.fit(logit, true)
                    model.aux_model["scaler"] = scaler
                scaler = model.aux_model["scaler"]
                prob = scaler.predict_proba(logit)
                prob = prob[:, 1] if logit.shape[1] <= 2 else prob
                val = auroc_scorer(true, prob, multi_class="ovr")
                logging_dict[f"{loader_name}-auroc"] = val

                if logit.shape[1] <= 2:
                    val = auprc_scorer(true, prob)
                else:
                    onehot = np.eye(logit.shape[1])[true]
                    val = auprc_scorer(onehot, prob)
                logging_dict[f"{loader_name}-auprc"] = val

                logging_dict[f"{loader_name}-raw-logit"] = logit
                logging_dict[f"{loader_name}-raw-true"] = true

                ## update class prior 
                if "train" in loader_name and loss_type in ["uPU", "nnPU"]:
                    curr_acc = logging_dict[f"{loader_name}-acc"]
                    curr_auroc = logging_dict[f"{loader_name}-auroc"]
                    ## compute positive rate of all samples
                    positive = label > 0
                    curr_prior = positive.astype(np.float32).sum() / label.shape[0]
                    logging_dict[f"{loader_name}-prior"] = best_prior
                    if curr_acc > best_acc and curr_auroc > best_auroc:
                        loss = update_loss(mode=loss_type, prior=curr_prior)
                        best_acc, best_auroc, best_prior = curr_acc, curr_auroc, curr_prior

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
        if contrastive_learning:
            centroid = intra_feat_sum / (intra_mem_num + 1e-10)
    
    return step_output, batch_indices

def training(
        num_epochs,
        split_path,
        scaler_path,
        num_node_features,
        model_dir,
        loss_type = "CE",
        logit_adjusted=False,
        BayesGNN=False,
        contrastive_learning=False
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
        "num_workers": 8, 
        "batch_size": 8,
    }
    dim_target_dict = {"CE": 2, "PN": 1, "uPU": 1, "nnPU": 1, "PCE": 1}
    positive_subgraph_dict = {"CE": False, "PN": False, "uPU": False, "nnPU": False, "PCE": True}
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": dim_target_dict[loss_type],
        "layers": [16, 16, 8], # [16, 16, 8]
        "dropout": 0.5,  #0.5
        "conv": "GINConv",
    }
    model_name = loss_type + "CL" if contrastive_learning else loss_type
    model_dir = model_dir / f"BayesGIN_{model_name}" if BayesGNN else model_dir / f"GIN_{model_name}"
    optim_kwargs = {
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,  # 1.0e-4
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
            BayesGNN=BayesGNN,
            arch_kwargs=arch_kwargs,
            loader_kwargs=loader_kwargs,
            optim_kwargs=optim_kwargs,
            preproc_func=node_scaler.transform,
            loss_type=loss_type,
            logit_adjusted=logit_adjusted,
            positive_subgraph=positive_subgraph_dict[loss_type],
            contrastive_learning=contrastive_learning
        )
    return

def inference(
        split_path,
        scaler_path,
        num_node_features,
        pretrained_dir,
        select_positive_samples=False,
        positive_unlabeled=False
):
    """node classification 
    """
    splits = joblib.load(split_path)
    node_scaler = joblib.load(scaler_path)
    
    loader_kwargs = {
        "num_workers": 8,
        "batch_size": 8,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 3,
        "layers": [16, 16, 8],
        "dropout": 0.5,
        "conv": "GINConv"
    }
    pretrained_PU_dir = pretrained_dir / "GIN_nnPU"
    pretrained_dir = pretrained_dir / "GIN_PCE"
    cum_stats = []
    for split_idx, split in enumerate(splits):
        new_split = {"infer": [v[0] for v in split["test"]]}

        if positive_unlabeled:
            stat_files = recur_find_ext(f"{pretrained_PU_dir}/{split_idx:02d}/", [".json"])
            stat_files = [v for v in stat_files if ".old.json" not in v]
            assert len(stat_files) == 1
            chkpts, _ = select_checkpoints(
                stat_files[0],
                top_k=1,
                metric="infer-valid-A-auroc",
            )
            
            arch_kwargs_PU = arch_kwargs.copy()
            arch_kwargs_PU.update({"dim_target": 1})
            loader_kwargs_PU = {
                "num_workers": 1,
                "batch_size": 1,
            }
            cum_results_PU = []
            for chkpt_info in chkpts:
                chkpt_results, batch_indices = run_once(
                    new_split,
                    num_epochs=1,
                    save_dir=None,
                    pretrained=chkpt_info,
                    arch_kwargs=arch_kwargs_PU,
                    loader_kwargs=loader_kwargs_PU,
                    preproc_func=node_scaler.transform
                )
                chkpt_results, _ = list(zip(*chkpt_results))
                chkpt_results = np.array(chkpt_results).squeeze()
                positive_negative = np.zeros_like(chkpt_results)
                sigmoid = 1 / (1 + np.exp(-chkpt_results))
                positive_negative[sigmoid > 0.5] = 1
                positive_negative_list = []
                for start, end in zip(batch_indices[:-1], batch_indices[1:]):
                    positive_negative_list.append(positive_negative[start:end])
                cum_results_PU.append(positive_negative_list)

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
            positive_negative = cum_results_PU[i] if positive_unlabeled else None
            positive_subgraph = True if positive_unlabeled else False
            chkpt_results, _ = run_once(
                new_split,
                num_epochs=1,
                save_dir=None,
                pretrained=chkpt_info,
                arch_kwargs=arch_kwargs,
                loader_kwargs=loader_kwargs,
                preproc_func=node_scaler.transform,
                positive_subgraph=positive_subgraph,
                positive_negative=positive_negative
            )
            # * re-calibrate logit to probabilities
            model = SlideGraphArch(**arch_kwargs)
            model.load(*chkpt_info)
            scaler = model.aux_model["scaler"]
            chkpt_results, _ = list(zip(*chkpt_results))
            chkpt_results = np.array(chkpt_results).squeeze()
            chkpt_results = scaler.predict_proba(chkpt_results)

            cum_results.append(chkpt_results)
        cum_results = np.array(cum_results)
        cum_results = np.squeeze(cum_results)

        prob = cum_results
        if len(cum_results.shape) == 3:
            prob = np.mean(cum_results, axis=0)

        # * Calculate split statistics
        true_paths = [v[1] for v in split["test"]]
        true = [np.load(f"{path}") for path in true_paths]
        true = np.concatenate(true, axis=0)
        ## only compute metric for positive samples
        if select_positive_samples:
            positive = true.squeeze() > 0
            if arch_kwargs["dim_target"] == 4:
                prob = prob[positive, 1:]
            else:
                prob = prob[positive, 0:]
            prob = prob / prob.sum(axis=1, keepdims=True)
            true = true[positive] - 1
        onehot = np.eye(prob.shape[1])[true]

        ## compute per-class accuracy
        uids = np.unique(true)
        acc_scores = []
        for i in range(len(uids)):
            indices = true == uids[i]
            score = acc_scorer(true[indices], np.argmax(prob[indices], axis=1))
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
        "acc": np.stack(acc_list, axis=0).mean(axis=0),
        "auroc": np.stack(auroc_list, axis=0).mean(axis=0),
        "auprc": np.stack(auprc_list, axis=0).mean(axis=0)
    }
    print(f"Avg:", avg_stat)
    return cum_stats

def clustering(
        split_path,
        scaler_path,
        num_node_features,
        pretrained_dir
):
    """node classification 
    """
    splits = joblib.load(split_path)
    node_scaler = joblib.load(scaler_path)
    
    loader_kwargs = {
        "num_workers": 8,
        "batch_size": 8,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 3,
        "layers": [16, 16, 8],
        "dropout": 0.5,
        "conv": "GINConv"
    }
    pretrained_dir = pretrained_dir / "GIN_PCE"
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
        cum_results, cum_features = [], []
        for i, chkpt_info in enumerate(chkpts):
            chkpt_results, _ = run_once(
                new_split,
                num_epochs=1,
                save_dir=None,
                pretrained=chkpt_info,
                arch_kwargs=arch_kwargs,
                loader_kwargs=loader_kwargs,
                preproc_func=node_scaler.transform,
                positive_subgraph=False,
                positive_negative=None
            )
            # * re-calibrate logit to probabilities
            model = SlideGraphArch(**arch_kwargs)
            model.load(*chkpt_info)
            scaler = model.aux_model["scaler"]
            chkpt_results, chkpt_features = list(zip(*chkpt_results))
            chkpt_results = np.array(chkpt_results).squeeze()
            chkpt_results = scaler.predict_proba(chkpt_results)
            cum_results.append(chkpt_results)
            chkpt_features = np.array(chkpt_features).squeeze()
            cum_features.append(chkpt_features)
        cum_results = np.array(cum_results)
        cum_results = np.squeeze(cum_results)
        cum_features = np.array(cum_features)
        cum_features = np.squeeze(cum_features)

        prob = cum_results
        if len(cum_results.shape) == 3:
            prob = np.mean(cum_results, axis=0)
        pred = prob.argmax(axis=1)
        ## k-means clustering for positive predictions
        positive = pred > 0
        pos_features = cum_features[positive]
        kmeans = KMeans(n_clusters=6, random_state=0, n_init="auto")
        pred[positive] = kmeans.labels_ + 1
        cum_stats.append([cum_features, pred])
    return cum_stats

def test(
        graph_path,
        label_path,
        scaler_path,
        num_node_features,
        pretrained,
        conv="MLP",
        BayesGNN=False,
        num_sampling=10
):
    """node classification 
    """
    node_scaler = joblib.load(scaler_path)
    
    loader_kwargs = {
        "num_workers": 1,
        "batch_size": 1,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 2,
        "layers": [16, 16, 8],
        "dropout": 0.5,
        "conv": conv
    }
    
    new_split = {"infer": [graph_path] * num_sampling} if BayesGNN else {"infer": [graph_path]}
    outputs, _ = run_once(
        new_split,
        num_epochs=1,
        save_dir=None,
        BayesGNN=BayesGNN,
        pretrained=pretrained,
        arch_kwargs=arch_kwargs,
        loader_kwargs=loader_kwargs,
        preproc_func=node_scaler.transform,
    )

    # * re-calibrate logit to probabilities
    outputs = list(zip(*outputs))
    outputs = np.array(outputs).squeeze()
    if arch_kwargs["dim_target"] == 1:
        label = np.zeros_like(outputs)
        sigmoid = 1 / (1 + np.exp(-outputs))
        label[sigmoid > 0.5] = 1
        prob = np.eye(2)[label.astype(np.int32)]
    else:
        if BayesGNN:
            model = SlideBayesGraphArch(**arch_kwargs)
        else:
            model = SlideGraphArch(**arch_kwargs)
        model.load(*pretrained)
        scaler = model.aux_model["scaler"]
        prob = scaler.predict_proba(outputs)
    if BayesGNN:
        step = prob.shape[0] // num_sampling
        prob = np.array([prob[i*step:(i+1)*step] for i in range(num_sampling)])
    # true = np.load(f"{label_path}").astype(np.uint32)
    # true = np.array(true, np.int32).squeeze()
    # binary = np.zeros_like(true)
    # binary[true > 0] = 1
    # true_prob = np.zeros((len(true), 2), np.float32)
    # true_prob[binary] = prob[true]
    # auroc = auroc_scorer(true, true_prob, multi_class="ovr")
    # onehot = np.eye(2)[binary]
    # auprc = auprc_scorer(onehot, true_prob)
    # logging.info(f"AUROC: {auroc:0.5f}, AUPRC: {auprc:0.5f}")
    return prob
    
def compute_label_portion(split_path):
    splits = joblib.load(split_path)
    for idx, split in enumerate(splits):
        for name, subset in split.items():
            num_nodes = 0
            num_freqs = np.zeros([2], np.uint32)
            for _, path in subset:
                label = np.array(np.load(pathlib.Path(path)), np.uint32)
                num_nodes += len(label)
                uids, freqs = np.unique(label, return_counts=True)
                num_freqs[uids] = num_freqs[uids] + freqs
            p = (num_freqs / num_nodes).tolist()
            logging.info(f"Fold-{idx}: nodes={num_nodes}; {name}-portion=[{p[0]:0.5f}, {p[1]:0.5f}]")
    return 




if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', default="/well/rittscher/shared/datasets/KiBla/cases")
    parser.add_argument('--wsi_ann_dir', default="a_06semantic_segmentation/wsi_kidney_annotations")
    parser.add_argument('--save_dir', default="a_06semantic_segmentation", type=str)
    parser.add_argument('--mask_method', default='otsu', choices=["otsu", "morphological"], help='method of tissue masking')
    parser.add_argument('--task', default="kidney", choices=["bladder", "kidney"], type=str)
    parser.add_argument('--mode', default="tile", choices=["tile", "wsi"], type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--feature_mode', default="vit", choices=["cnn", "finetuned_cnn", "vit"], type=str)
    parser.add_argument('--node_features', default=384, choices=[2048, 2048, 384], type=int)
    parser.add_argument('--resolution', default=0.25, type=float)
    parser.add_argument('--units', default="mpp", type=str)
    parser.add_argument('--loss', default="CE", choices=["CE", "PN", "uPU", "nnPU", "PCE"], type=str)
    parser.add_argument('--Bayes', default=False, type=bool, help="whether to build Bayesian GNN")
    args = parser.parse_args()

    ## select annotated wsi
    wsi_dir = pathlib.Path(args.wsi_dir)
    wsi_ann_dir = pathlib.Path(args.wsi_ann_dir)
    wsi_paths, wsi_ann_paths = select_wsi_annotated(wsi_dir, wsi_ann_dir)
    logging.info("Totally {} wsi and {} annotation!".format(len(wsi_paths), len(wsi_ann_paths)))
    
    ## set save dir
    save_tile_dir = pathlib.Path(f"{args.save_dir}/wsi_{args.task}_tiles")
    save_msk_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_masks")
    save_feature_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_features/{args.feature_mode}")
    save_label_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_labels/{args.feature_mode}")
    save_model_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_models/{args.feature_mode}")
    

    ## define label name and value, should be consistent with annotation
    lab_dict = {
        "Background": 0,
        "Tumour areas": 1,
    }
    ## set annotation resolution

    ## generate ROI tile from wsi based on annotation
    if args.mode == "tile":
        # generate_tile_from_wsi(
        #     wsi_paths=wsi_paths,
        #     wsi_ann_paths=wsi_ann_paths,
        #     lab_dict=lab_dict,
        #     save_tile_dir=save_tile_dir,
        #     tile_size=10240,
        #     resolution=args.resolution,
        #     units=args.units,
        # )
        wsi_paths = sorted(save_tile_dir.glob("*.jpg"))
        wsi_ann_paths = sorted(save_tile_dir.glob("*.json"))
        logging.info("Totally {} tile and {} annotation!".format(len(wsi_paths), len(wsi_ann_paths)))

    ## generate wsi tissue mask
    # if args.mode == "wsi":
    #     generate_wsi_tissue_mask(wsi_paths, save_msk_dir, args.mask_method)

    ## extract wsi feature
    # if args.mode == "wsi":
    #     save_msk_paths = sorted(save_msk_dir.glob("*.jpg"))
    # else:
    #     save_msk_paths = None
    # extract_wsi_feature(
    #     wsi_paths=wsi_paths,
    #     wsi_msk_paths=save_msk_paths,
    #     feature_mode=args.feature_mode,
    #     save_dir=save_feature_dir,
    #     mode=args.mode,
    #     resolution=args.resolution,
    #     units=args.units,
    # )

    ## construct wsi graph
    # construct_wsi_graph(
    #     wsi_paths=wsi_paths,
    #     save_dir=save_feature_dir,
    #     n_jobs=1 if args.mode == "wsi" else 8
    # )

    # ## generate node label from annotation
    # wsi_graph_paths = sorted(save_feature_dir.glob("*.json")) 
    # node_size = {"cnn": 224, "fintuned_cnn": 224, "vit": 256}[args.feature_mode]  
    # generate_node_label(
    #     wsi_paths=wsi_paths,
    #     wsi_annot_paths=wsi_ann_paths,
    #     wsi_graph_paths=wsi_graph_paths,
    #     lab_dict=lab_dict,
    #     save_lab_dir=save_label_dir,
    #     node_size=node_size,
    #     min_ann_ratio=0.1,
    #     resolution=args.resolution,
    #     units=args.units,
    # )

    ## split data set
    # num_folds = 5
    # test_ratio = 0.2
    # train_ratio = 0.8 * 0.9
    # valid_ratio = 0.8 * 0.1
    # wsi_graph_paths = sorted(save_feature_dir.glob("*.json"))
    # wsi_label_paths = sorted(save_label_dir.glob("*.label.npy"))
    # splits = generate_data_split(
    #     x=wsi_graph_paths,
    #     y=wsi_label_paths,
    #     train=train_ratio,
    #     valid=valid_ratio,
    #     test=test_ratio,
    #     num_folds=num_folds,
    # )
    # num_train = len(splits[0]["train"])
    # logging.info(f"Number of training samples: {num_train}.")
    # num_valid = len(splits[0]["valid"])
    # logging.info(f"Number of validating samples: {num_valid}.")
    # num_test = len(splits[0]["test"])
    # logging.info(f"Number of testing samples: {num_test}.")
    # mkdir(save_model_dir)
    split_path = f"{save_model_dir}/splits.dat"
    # joblib.dump(splits, split_path)
    # compute_label_portion(split_path)

    ## compute mean and std on training data for normalization 
    # splits = joblib.load(split_path)
    # train_wsi_paths = [path for path, _ in splits[0]["train"]]
    # loader = SlideGraphDataset(train_wsi_paths, mode="infer")
    # loader = DataLoader(
    #     loader,
    #     num_workers=8,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    # )
    # node_features = [v.x.numpy() for v in tqdm(loader)]
    # node_features = np.concatenate(node_features, axis=0)
    # node_scaler = StandardScaler(copy=False)
    # node_scaler.fit(node_features)
    scaler_path = f"{save_model_dir}/node_scaler.dat"
    # joblib.dump(node_scaler, scaler_path)

    ## training
    # training(
    #     num_epochs=args.epochs,
    #     split_path=split_path,
    #     scaler_path=scaler_path,
    #     num_node_features=args.node_features,
    #     model_dir=save_model_dir,
    #     loss_type=args.loss,
    #     logit_adjusted=False,
    #     BayesGNN=args.Bayes,
    #     contrastive_learning=True
    # )

    # ## inference
    inference(
        split_path=split_path,
        scaler_path=scaler_path,
        num_node_features=args.node_features,
        pretrained_dir=save_model_dir,
        select_positive_samples=True,
        positive_unlabeled=True
    )

    ## clustering
    clusters = clustering(
        split_path=split_path,
        scaler_path=scaler_path,
        num_node_features=args.node_features,
        pretrained_dir=save_model_dir,
    )
    graph_feature_visualization(
        wsi_paths=None,
        save_graph_dir=None,
        save_label_dir=None,
        num_class=7,
        features=clusters[0][0],
        colors=clusters[0][1]
    )

    ## visualize feature
    # graph_feature_visualization(
    #     wsi_paths=wsi_paths[0:10],
    #     save_graph_dir=save_feature_dir,
    #     save_label_dir=save_label_dir,
    #     num_class=len(lab_dict)
    # )

    ## visualize prediction
    # fold = 0
    # split = joblib.load(split_path)[fold]
    # graph_paths = [x for x, _ in split["test"]]
    # wsi_name = pathlib.Path(graph_paths[7]).stem # 0,2,4,7
    # wsi_path = save_tile_dir / f"{wsi_name}.jpg"
    # graph_path = save_feature_dir / f"{wsi_name}.json"
    # label_path = save_label_dir / f"{wsi_name}.label.npy"
    # pretrained_model = f"a_06semantic_segmentation/tile_kidney_models/vit/BayesGIN_CE/{fold:02d}/epoch=049.weights.pth"
    # pretrained_aux_model = f"a_06semantic_segmentation/tile_kidney_models/vit/BayesGIN_CE/{fold:02d}/epoch=049.aux.dat"
    # prob = test(
    #     graph_path=graph_path,
    #     label_path=label_path,
    #     scaler_path=scaler_path,
    #     num_node_features=args.node_features,
    #     pretrained=[pretrained_model, pretrained_aux_model],
    #     conv="GINConv",
    #     BayesGNN=args.Bayes
    # )
    # visualize_graph(
    #     wsi_path=wsi_path,
    #     graph_path=graph_path,
    #     label=prob,
    #     positive_graph=False,
    #     resolution=args.resolution,
    #     units=args.units
    # )





    



