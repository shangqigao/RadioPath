import sys
sys.path.append('../')

import pathlib
import logging
import copy
import torch
import joblib
import argparse
import yaml
import json
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

import torchbnn as bnn

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import accuracy_score as acc_scorer
from torch_geometric.loader import DataLoader

from tiatoolbox.utils.misc import save_as_json
from tiatoolbox import logger

from common.m_utils import mkdir, create_pbar, load_json, rm_n_mkdir, recur_find_ext
from common.m_utils import reset_logging, select_checkpoints, select_wsi

from models.a_06generative_SR.m_generative_diffusion import SlideGraphSpectrumDataset
from models.a_06generative_SR.m_generative_diffusion import SlideGraphSpectrumDiffusionArch
from models.a_06generative_SR.m_generative_diffusion import ScalarMovingAverage
from tiatoolbox.models.architecture.gsdm.utils.loader import load_model, load_model_params, load_loss_fn2
from tiatoolbox.models.architecture.gsdm.utils.loader import load_eval_settings, load_sampling_fn2
from tiatoolbox.models.architecture.gsdm.utils.graph_utils import adjs_to_graphs
from tiatoolbox.models.architecture.gsdm.evaluation.stats import eval_graph_list

torch.multiprocessing.set_sharing_strategy("file_system")

import warnings
warnings.filterwarnings('ignore')

def generate_data_split(
        x: list,
        train: float,
        valid: float,
        test: float,
        num_folds: int,
        seed: int = 5,
):
    """Helper to generate splits
    Args:
        x (list): a list of image paths
        train (float): ratio of training samples
        valid (float): ratio of validating samples
        test (float): ratio of testing samples
        num_folds (int): number of folds for cross-validation
        seed (int): random seed
    Returns:
        splits (list): a list of folds, each fold consists of train, valid, and test splits
    """
    assert train + valid + test - 1.0 < 1.0e-10, "Ratios must sum to 1.0"

    outer_splitter = ShuffleSplit(
        n_splits=num_folds,
        train_size=train + valid,
        random_state=seed,
    )
    inner_splitter = ShuffleSplit(
        n_splits=1,
        train_size=train / (train + valid),
        random_state=seed,
    )

    splits = []
    for train_valid_idx, test_idx in outer_splitter.split(x):
        test_x = [x[idx] for idx in test_idx]
        x_ = [x[idx] for idx in train_valid_idx]

        train_idx, valid_idx = next(iter(inner_splitter.split(x_)))
        valid_x = [x_[idx] for idx in valid_idx]
        train_x = [x_[idx] for idx in train_idx]

        assert len(set(train_x).intersection(set(valid_x))) == 0
        assert len(set(valid_x).intersection(set(test_x))) == 0
        assert len(set(train_x).intersection(set(test_x))) == 0

        splits.append(
            {
                "train": train_x,
                "valid": valid_x,
                "test": test_x,
            }
        )
    return splits

def run_once(
        dataset_dict,
        save_dir,
        config,
        on_gpu=True,
        preproc_func=None,
        pretrained=None,
        loader_kwargs=None
):
    """running the inference or training loop once"""
    if loader_kwargs is None:
        loader_kwargs = {}
 
    params_x, params_adj = load_model_params(config)
    model_x, model_adj = load_model(params_x), load_model(params_adj)
    model = SlideGraphSpectrumDiffusionArch(config, params_x, params_adj, model_x, model_adj)
    if pretrained is not None:
        model.load(*pretrained)
    model = model.to("cuda")

    optimizer_x = torch.optim.Adam(model.model_x.parameters(), lr=config.train.lr, 
                                   weight_decay=config.train.weight_decay)
    scheduler_x = torch.optim.lr_scheduler.ExponentialLR(optimizer_x, gamma=config.train.lr_decay)

    optimizer_adj = torch.optim.Adam(model.model_adj.parameters(), lr=config.train.lr, 
                                     weight_decay=config.train.weight_decay)  
    scheduler_adj = torch.optim.lr_scheduler.ExponentialLR(optimizer_adj, gamma=config.train.lr_decay)                             
    
    ## Set loss
    loss = load_loss_fn2(config)

    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        ds = SlideGraphSpectrumDataset(
            subset, 
            mode=subset_name, 
            preproc=preproc_func, 
        )
        loader_dict[subset_name] = DataLoader(
            ds,
            drop_last=subset_name == "train",
            shuffle=subset_name == "train",
            **_loader_kwargs,
        )

    for epoch in range(config.train.num_epochs):
        logger.info("EPOCH: %03d", epoch)
        for loader_name, loader in loader_dict.items():
            step_output = []
            ema = ScalarMovingAverage()
            pbar = create_pbar(loader_name, len(loader))
            for step, batch_data in enumerate(loader):
                if loader_name == "train":
                    output = model.train_batch(model, batch_data, on_gpu, loss, (optimizer_x, optimizer_adj))
                    ema({"loss": output[0] + output[1], "loss_x": output[0], "loss_adj": output[1]})
                    pbar.postfix[1]["step"] = step
                    pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
                else:
                    output = model.infer_batch(model, batch_data, on_gpu, loss)
                    step_output.append(output)
                pbar.update()
            pbar.close()

            logging_dict = {}
            if loader_name == "train":
                for val_name, val in ema.tracking_dict.items():
                    logging_dict[f"train-EMA-{val_name}"] = val
            elif "infer" in loader_name and any(v in loader_name for v in ["train", "valid"]):
                output = list(zip(*step_output))
                loss_x, loss_adj = output
                avg_loss_x = sum(loss_x) / len(loss_x)
                avg_loss_adj = sum(loss_adj) / len(loss_adj)
                
                logging_dict[f"{loader_name}-avg_loss_x"] = avg_loss_x
                logging_dict[f"{loader_name}-avg_loss_adj"] = avg_loss_adj

            for val_name, val in logging_dict.items():
                if "raw" not in val_name:
                    logging.info("%s: %0.5f\n", val_name, val)
            
            if "train" not in loader_dict:
                continue

            if (epoch + 1) % 500 == 0:
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
                    f"{save_dir}/epoch={epoch:03d}.model_x.pth",
                    f"{save_dir}/epoch={epoch:03d}.model_adj.pth",
                )
        scheduler_x.step()
        scheduler_adj.step()
    
    return step_output


def training(
        split_path,
        scaler_path,
        config_path,
        model_dir,
        pretrained=None,
        beta_type="linear",
        seed=42
):
    """train node classification neural networks
    Args:
        split_path (str): the path of storing data splits
        scaler_path (str): the path of storing data normalization
        num_node_features (int): the dimension of node feature
        model_dir (str): directory of saving models
    """
    splits = joblib.load(split_path)
    node_scaler = joblib.load(scaler_path)
    config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))
    config.type = beta_type
    config.seed = seed
    
    loader_kwargs = {
        "num_workers": 8, 
        "batch_size": config.data.batch_size,
    }
    
    model_dir = model_dir / "Diffusion_Prior_GSDM"
    for split_idx, split in enumerate(splits):
        new_split = {
            "train": split["train"],
            "infer-valid-A": split["valid"],
            "infer-valid-B": split["test"],
        }
        split_save_dir = pathlib.Path(f"{model_dir}/{split_idx:02d}/")
        rm_n_mkdir(split_save_dir)
        reset_logging(split_save_dir)
        run_once(
            new_split,
            save_dir=split_save_dir,
            config=config,
            preproc_func=node_scaler.transform,
            pretrained=pretrained,
            loader_kwargs=loader_kwargs,
        )
    return

def sampling(model, batch_data, on_gpu, sampling, train_edge_index, thr=0.5, eps=1e-5):
    device = "cuda" if on_gpu else "cpu"
    infer_graphs = batch_data.to(device)
    infer_graphs.x = infer_graphs.x.type(torch.float32)
    infer_adj = to_dense_adj(infer_graphs.edge_index)

    train_edge_index = train_edge_index.to(device)
    max_num_nodes = model.module.config.data.max_num_nodes
    train_adj = to_dense_adj(train_edge_index, max_num_nodes=max_num_nodes)
    flags = torch.abs(train_adj).sum(-1).gt(eps)

    model.module.model_x.eval()
    model.module.model_adj.eval()
    with torch.inference_mode():
        pred_x, pred_adj, _ = sampling(model.module.model_x, model.module.model_adj, flags, train_adj)
    
    # mask adjacency matrix with flags
    if len(pred_adj.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    pred_adj = pred_adj * flags.unsqueeze(-1)
    pred_adj = pred_adj * flags.unsqueeze(-2)

    pred_adj = pred_adj.triu(1)
    pred_adj = pred_adj + torch.transpose(pred_adj, -1, -2)

    # quantize adjacency matrix
    pred_adj = torch.where(pred_adj < thr, torch.zeros_like(pred_adj), torch.ones_like(pred_adj))

    infer_adj = infer_adj.detach().cpu().numpy()
    pred_adj = pred_adj.detach().cpu().numpy()
    infer_x = infer_graphs.x.detach().cpu().numpy()
    pred_x = pred_x.detach().cpu().numpy()

    return [infer_x, pred_x, infer_adj, pred_adj]

def inference(
        split_path,
        scaler_path,
        num_node_features,
        pretrained_dir,
        select_positive_samples=False,
        select_low_high_samples=False,
        PN_learning=False,
        PU_learning=False,
        BayesGNN=False
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
        "dim_target": 1,
        "layers": [16, 16, 8],
        "dropout": 0.5,
        "conv": "GINConv"
    }
    pretrained_PU_dir = pretrained_dir / "BayesGIN_nnPU"
    pretrained_dir = pretrained_dir / "BayesGIN_nnPU"
    cum_stats = []
    for split_idx, split in enumerate(splits):
        new_split = {"infer": [v[0] for v in split["test"]]}

        if PN_learning or PU_learning:
            stat_files = recur_find_ext(f"{pretrained_PU_dir}/{split_idx:02d}/", [".json"])
            stat_files = [v for v in stat_files if ".old.json" not in v]
            assert len(stat_files) == 1
            chkpts, _ = select_checkpoints(
                stat_files[0],
                top_k=1,
                metric="infer-valid-A-auroc",
            )
            
            arch_kwargs_PU = arch_kwargs.copy()
            if PN_learning:
                arch_kwargs_PU.update({"dim_target": 2})
            else:
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
                    BayesGNN=BayesGNN,
                    pretrained=chkpt_info,
                    arch_kwargs=arch_kwargs_PU,
                    loader_kwargs=loader_kwargs_PU,
                    preproc_func=node_scaler.transform
                )
                chkpt_results = list(zip(*chkpt_results))
                chkpt_results = np.array(chkpt_results).squeeze()
                if PN_learning:
                    if BayesGNN:
                        model = SlideGraphSpectrumDiffusionArch(**arch_kwargs)
                    else:
                        model = SlideGraphSpectrumDiffusionArch(**arch_kwargs)
                    model.load(*chkpt_info)
                    scaler = model.aux_model["scaler"]
                    prob = scaler.predict_proba(chkpt_results)
                    positive_negative = np.argmax(prob, axis=1)
                else:
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
        if BayesGNN:
            new_split = {"infer": [v[0] for v in split["test"]]*10}

        cum_results, cum_sigmoid = [], []
        for i, chkpt_info in enumerate(chkpts):
            positive_negative = cum_results_PU[i] if PN_learning or PU_learning else None
            if positive_negative is not None and BayesGNN:
                positive_negative = positive_negative*10
            positive_subgraph = True if PN_learning or PU_learning else False
            # true_paths = [v[1] for v in split["test"]]*1
            # positive_negative = [np.load(f"{path}") for path in true_paths]
            # positive_subgraph = True
            chkpt_results, _ = run_once(
                new_split,
                num_epochs=1,
                save_dir=None,
                BayesGNN=BayesGNN,
                pretrained=chkpt_info,
                arch_kwargs=arch_kwargs,
                loader_kwargs=loader_kwargs,
                preproc_func=node_scaler.transform,
                positive_subgraph=positive_subgraph,
                positive_negative=positive_negative
            )
            # * re-calibrate logit to probabilities
            if BayesGNN:
                model = SlideGraphSpectrumDiffusionArch(**arch_kwargs)
            else:
                model = SlideGraphSpectrumDiffusionArch(**arch_kwargs)
            model.load(*chkpt_info)
            scaler = model.aux_model["scaler"]
            chkpt_results = list(zip(*chkpt_results))
            chkpt_results = np.array(chkpt_results).squeeze()
            if arch_kwargs["dim_target"] == 1:
                chkpt_results = chkpt_results.reshape(-1, 1)
                sigmoid = 1 / (1 + np.exp(-chkpt_results))
                cum_sigmoid.append(sigmoid)
            chkpt_results = scaler.predict_proba(chkpt_results)

            cum_results.append(chkpt_results)
        cum_results = np.array(cum_results)
        cum_results = np.squeeze(cum_results)
        if arch_kwargs["dim_target"] == 1:
            cum_sigmoid = np.array(cum_sigmoid)
            cum_sigmoid = np.squeeze(cum_sigmoid)
            if len(cum_sigmoid.shape) == 2:
                sigmoid = np.mean(cum_sigmoid, axis=0)
            else:
                sigmoid = cum_sigmoid

            # model ensembling if Bayesian neural networks
            if BayesGNN:
                N = sigmoid.shape[0] // 10
                sigmoid = np.array([sigmoid[i*N:(i+1)*N] for i in range(10)]).mean(axis=0)

        prob = cum_results
        if len(cum_results.shape) == 3:
            prob = np.mean(cum_results, axis=0)

        # model ensembling if Bayesian neural networks
        if BayesGNN:
            N = prob.shape[0] // 10
            prob = np.array([prob[i*N:(i+1)*N] for i in range(10)]).mean(axis=0)

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
        elif select_low_high_samples:
            assert arch_kwargs["dim_target"] == 2
            low = true.squeeze() == 1
            high = true.squeeze() == 3
            prob = np.concatenate([prob[low], prob[high]], axis=0)
            true = np.concatenate([true[low] - 1, true[high] - 2], axis=0)
            onehot = true
        else:
            if arch_kwargs["dim_target"] <= 2:
                true = (true > 0).astype(np.int32)
                onehot = true
            else:
                onehot = np.eye(prob.shape[1])[true] 

        ## compute per-class accuracy
        if arch_kwargs["dim_target"] == 1:
            pred = (sigmoid > 0.5).astype(np.int32)
        else:
            pred = np.argmax(prob, axis=1)
        uids = np.unique(true)
        acc_scores = []
        for i in range(len(uids)):
            indices = true == uids[i]
            score = acc_scorer(true[indices], pred[indices])
            acc_scores.append(score)

        prob = prob[:, 1] if arch_kwargs["dim_target"] <= 2 else prob
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

def test(
        graph_path,
        scaler_path,
        num_node_features,
        pretrained_classificaiton,
        pretrained_detection=None,
        conv="MLP",
        PN_learning=False,
        PU_learning=False,
        BayesGNN=False,
        num_sampling=10,
        borderline=False
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

    # BayesGNN = True
    new_split = {"infer": [graph_path] * num_sampling} if BayesGNN else {"infer": [graph_path]}
    if PN_learning or PU_learning:
        assert pretrained_detection is not None
        arch_kwargs_PU = arch_kwargs.copy()
        if PN_learning:
            arch_kwargs_PU.update({"dim_target": 2})
        else:
            arch_kwargs_PU.update({"dim_target": 1})
        chkpt_results, _ = run_once(
            new_split,
            num_epochs=1,
            save_dir=None,
            BayesGNN=BayesGNN,
            pretrained=pretrained_detection,
            arch_kwargs=arch_kwargs_PU,
            loader_kwargs=loader_kwargs,
            preproc_func=node_scaler.transform,
        )
        chkpt_results = list(zip(*chkpt_results))
        chkpt_results = np.array(chkpt_results).squeeze()
        if PN_learning:
            if BayesGNN:
                model = SlideGraphSpectrumDiffusionArch(**arch_kwargs)
            else:
                model = SlideGraphSpectrumDiffusionArch(**arch_kwargs)
            model.load(*pretrained_detection)
            scaler = model.aux_model["scaler"]
            prob = scaler.predict_proba(chkpt_results)
            if BayesGNN:
                step = prob.shape[0] // num_sampling
                PN_std = np.array([prob[i*step:(i+1)*step] for i in range(num_sampling)]).std(axis=0)
                prob = np.array([prob[i*step:(i+1)*step] for i in range(num_sampling)]).mean(axis=0)
            positive_negative = np.argmax(prob, axis=1)
        else:
            sigmoid = 1 / (1 + np.exp(-chkpt_results))
            PN_std = np.zeros_like(sigmoid)
            if BayesGNN:
                step = sigmoid.shape[0] // num_sampling
                PN_std = np.array([sigmoid[i*step:(i+1)*step] for i in range(num_sampling)]).std(axis=0)
                sigmoid = np.array([sigmoid[i*step:(i+1)*step] for i in range(num_sampling)]).mean(axis=0)
            positive_negative = np.zeros_like(sigmoid)
            positive_negative[sigmoid > 0.5] = 1
        positive_subgraph = True
    else:
        positive_negative = None
        positive_subgraph = False
        PN_std = None
    
    # BayesGNN = False
    # new_split = {"infer": [graph_path] * num_sampling} if BayesGNN else {"infer": [graph_path]}
    positive_negative = [positive_negative]*num_sampling if BayesGNN else [positive_negative]
    PN_std = [PN_std]*num_sampling if BayesGNN else [PN_std]
    outputs, _ = run_once(
        new_split,
        num_epochs=1,
        save_dir=None,
        BayesGNN=BayesGNN,
        pretrained=pretrained_classificaiton,
        arch_kwargs=arch_kwargs,
        loader_kwargs=loader_kwargs,
        preproc_func=node_scaler.transform,
        positive_subgraph=positive_subgraph,
        positive_negative=positive_negative
    )

    # * re-calibrate logit to probabilities
    outputs = list(zip(*outputs))
    outputs = np.array(outputs).squeeze()
    if arch_kwargs["dim_target"] == 1:
        sigmoid = 1 / (1 + np.exp(-outputs))
        prob = np.zeros((sigmoid.shape[0], 2))
        prob[:, 0] = 1 - sigmoid
        prob[:, 1] = sigmoid
    else:
        if BayesGNN:
            model = SlideGraphSpectrumDiffusionArch(**arch_kwargs)
        else:
            model = SlideGraphSpectrumDiffusionArch(**arch_kwargs)
        model.load(*pretrained_classificaiton)
        scaler = model.aux_model["scaler"]
        prob = scaler.predict_proba(outputs)
        if pretrained_detection is not None:
            positive_negative = np.concatenate(positive_negative, axis=0)
            PN_std = np.concatenate(PN_std, axis=0)
            PN_prob = np.zeros([positive_negative.shape[0], 4])
            positive = positive_negative > 0
            if borderline:
                certain = PN_std < 0.1
                positive = certain & positive
            negative = np.logical_not(positive)
            PN_prob[negative, 0] = 1
            if prob.shape[1] == 3:
                PN_prob[positive, 1:] = prob[positive, :]
            elif prob.shape[1] == 2:
                PN_prob[positive, 1] = prob[positive, 0]
                PN_prob[positive, 3] = prob[positive, 1]
            prob = PN_prob
    if BayesGNN:
        step = prob.shape[0] // num_sampling
        prob = np.array([prob[i*step:(i+1)*step] for i in range(num_sampling)])
        if borderline:
            mean, std = prob.mean(axis=0), prob.std(axis=0)
            label = np.argmax(mean, axis=1)
            uncertainty = np.array([std[i, label[i]] for i in range(std.shape[0])])
            prob = np.zeros_like(mean)
            prob[:, 2] = (uncertainty > 0.2).astype(np.float32)

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

def node_statistics(wsi_graph_paths):
    node_features = []
    for graph_path in wsi_graph_paths:
        with pathlib.Path(graph_path).open() as fptr:
            graph_dict = json.load(fptr)
        node_features.append(np.array(graph_dict["x"]))
    num_nodes = [len(n) for n in node_features]
    max_num = max(num_nodes)
    min_num = min(num_nodes)
    logging.info(f"Max node num: {max_num}, Min node num: {min_num}")
    return

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', default="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/TCGA/WSI")
    parser.add_argument('--dataset', default="TCGA-RCC", type=str)
    parser.add_argument('--config', default="./a_06generative_SR/TCGA_graphs.yaml", type=str)
    parser.add_argument('--save_pathomics_dir', default="/home/sg2162/rds/hpc-work/Experiments/pathomics", type=str)
    parser.add_argument('--mode', default="wsi", choices=["tile", "wsi"], type=str)
    parser.add_argument('--pathomics_mode', default="uni", choices=["cnn", "vit", "uni", "conch", "chief"], type=str)
    parser.add_argument('--type', default="linear", choices=["linear", "exp", "cosine", "tanh"], type=str)
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
    save_model_dir = pathlib.Path(f"{args.save_pathomics_dir}/{args.dataset}_{args.mode}_models/{args.pathomics_mode}")

    ## node statistics
    wsi_graph_paths = [save_pathomics_dir / f"{p.stem}.MST.json" for p in wsi_paths]
    # node_statistics(wsi_graph_paths)

    ## split data set
    num_folds = 5
    test_ratio = 0.2
    train_ratio = 0.8 * 0.9
    valid_ratio = 0.8 * 0.1
    splits = generate_data_split(
        x=wsi_graph_paths,
        train=train_ratio,
        valid=valid_ratio,
        test=test_ratio,
        num_folds=num_folds,
    )
    mkdir(save_model_dir)
    split_path = f"{save_model_dir}/diffusion_splits.dat"
    joblib.dump(splits, split_path)
    splits = joblib.load(split_path)
    num_train = len(splits[0]["train"])
    logging.info(f"Number of training samples: {num_train}.")
    num_valid = len(splits[0]["valid"])
    logging.info(f"Number of validating samples: {num_valid}.")
    num_test = len(splits[0]["test"])
    logging.info(f"Number of testing samples: {num_test}.")

    ## compute mean and std on training data for normalization 
    splits = joblib.load(split_path)
    train_wsi_paths = [path for path in splits[0]["train"]]
    loader = SlideGraphSpectrumDataset(train_wsi_paths, mode="infer")
    loader = DataLoader(
        loader,
        num_workers=8,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    node_features = [v.x.numpy() for v in loader]
    num_nodes = [len(f) for f in node_features]

    node_features = np.concatenate(node_features, axis=0)
    node_scaler = StandardScaler(copy=False)
    node_scaler.fit(node_features)
    scaler_path = f"{save_model_dir}/diffusion_node_scaler.dat"
    joblib.dump(node_scaler, scaler_path)

    ## training
    training(
        split_path=split_path,
        scaler_path=scaler_path,
        config_path=args.config,
        model_dir=save_model_dir,
        beta_type=args.type,
        seed=42
    )

    # ## inference
    # inference(
    #     split_path=split_path,
    #     scaler_path=scaler_path,
    #     num_node_features=args.node_features,
    #     pretrained_dir=save_model_dir,
    #     select_positive_samples=False,
    #     select_low_high_samples=False,
    #     PN_learning=False,
    #     PU_learning=False,
    #     BayesGNN=args.Bayes
    # )