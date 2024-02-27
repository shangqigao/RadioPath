import sys
sys.path.append('../')

import csv
import pathlib
import joblib
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

csv.field_size_limit(sys.maxsize)

from common.m_utils import mkdir

from models.a_02tissue_masking.m_tissue_masking import generate_wsi_tissue_mask
from models.a_04feature_extraction.m_feature_extraction import extract_wsi_feature
from models.a_05feature_aggregation.m_graph_construction import construct_wsi_graph
from models.a_06semantic_segmentation.m_bladder_carcinoma_classification import test

def read_clinical_outcome(m_path, o_path, colnames):
    mapping = {}
    with open(m_path) as mfile:
        m_reader = pd.read_csv(mfile)
        for idx, row in m_reader.iterrows():
            snum = row["Study Number"]
            hnum = str(row["Histology Number"]).replace("/", "_")
            mapping.update({snum: hnum})
    # noT, doT = {}, {}
    outcome = {}
    with open(o_path) as ofile:
        o_reader = pd.read_csv(ofile)
        fil = [
            "Length of time to death", 
            "Adjuvant Treatment", 
            "Treatment for NMIBC",
            "Treatment for MIBC"
        ]
        # sel = {
        #     "Treatment for NMIBC": "No",
        #     "if Yes - specify.1": "TURBT, BCG",
        #     # "if Yes - specify.2": "Cystectomy"
        # }
        # sel_k, sel_v = list(sel.keys()), list(sel.values())
        for idx, row in o_reader.iterrows():
            if all(pd.notnull(row[k]) for k in fil):
                snum = row["Study Number"]
                dict = {k: row[k] for k in colnames}
                outcome.update({mapping[snum]: dict})
    #             if any(row[k] == v for k, v in sel.items()):
    #                 snum = row["Study Number"]
    #                 dict = {k: row[k] for k in colnames}
    #                 if row[sel_k[0]] == "Yes":
    #                     dict.update({sel_k[0]: sel_v[1]})
    #                     doT.update({mapping[snum]: dict})
    #                 else:
    #                     noT.update({mapping[snum]: dict})
    # print(f"No treatment: {len(noT)} and Do treatment: {len(doT)}")
    # outcome = {}
    # noT_k, doT_k = list(noT.keys()), list(doT.keys())
    # for i in range(min(len(noT_k), len(doT_k))):
    #     k = noT_k[i]
    #     outcome.update({k: noT[k]})
    #     k = doT_k[i]
    #     outcome.update({k: doT[k]})
    return outcome

def compute_carcinoma_ratio(pred_dir, ids, threshold=0.3):
    cases = []
    for idx in ids:
        pred_paths = pathlib.Path(pred_dir).glob("*.pred.npy")
        wsis = []
        for path in pred_paths:
            if idx in path.stem:
                wsis.append(path)
        cases.append(wsis)

    def _compute_carcinoma_ratio_percase(idx, wsi_paths):
        preds = [np.load(f"{path}") for path in wsi_paths]
        ratios = []
        for pred in preds:
            pred = np.array(pred)
            assert pred.ndim == 3 and pred.shape[2] == 4
            pred_mean = np.mean(pred, axis=0)
            pred_lab = np.argmax(pred_mean, axis=-1)
            pred_std = np.std(pred, axis=0)
            num_nodes = pred_lab.shape[0]
            pred_map = np.array([pred_std[i, pred_lab[i]] for i in range(num_nodes)])
            borderline = (pred_lab > 0) & (pred_map > threshold)
            lowgrade = (pred_lab == 1) & (pred_map <= threshold)
            highgrade = (pred_lab == 3) & (pred_map <= threshold)
            b_ratio = borderline.astype(np.float32).sum() / num_nodes
            l_ratio = lowgrade.astype(np.float32).sum() / num_nodes
            h_ratio = highgrade.astype(np.float32).sum() / num_nodes
            ratios.append([l_ratio, b_ratio, h_ratio])
        id_ratio = [ids[idx], np.array(ratios).mean(axis=0)]
        return id_ratio
    
    # summarize cases in parallel
    summary = joblib.Parallel(n_jobs=8)(
        joblib.delayed(_compute_carcinoma_ratio_percase)(idx, case)
        for idx, case in enumerate(cases)
    ) 
    return dict(summary)

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="/well/rittscher/shared/datasets/KiBla/data")
    parser.add_argument('--wsi_dir', default="/well/rittscher/shared/datasets/KiBla/cases")
    parser.add_argument('--save_dir', default="a_07therapy_response", type=str)
    parser.add_argument('--mask_method', default='otsu', choices=["otsu", "morphological"], help='method of tissue masking')
    parser.add_argument('--task', default="bladder", choices=["bladder", "kidney"], type=str)
    parser.add_argument('--mode', default="wsi", choices=["tile", "wsi"], type=str)
    parser.add_argument('--feature_mode', default="vit", choices=["cnn", "finetuned_cnn", "vit"], type=str)
    parser.add_argument('--node_features', default=384, choices=[2048, 2048, 384], type=int)
    parser.add_argument('--resolution', default=0.25, type=float)
    parser.add_argument('--units', default="mpp", type=str)
    parser.add_argument('--loss', default="LHCE", choices=["CE", "PN", "uPU", "nnPU", "PCE", "LHCE"], type=str)
    parser.add_argument('--Bayes', default=True, type=bool, help="whether to build Bayesian GNN")
    args = parser.parse_args()

    # read clinical outcome
    m_path = pathlib.Path(args.data_dir) / "bladder_mapping_wsi.csv"
    o_path = pathlib.Path(args.data_dir) / "Bladder_Clinical_Outcome_Exported_Fileds_06_10_2023_11_48_54.csv"   
    names = [
        "Study Number", 
        "Length of time to death", 
        "Treatment for NMIBC", 
        "Treatment for MIBC", 
        "Adjuvant Treatment"
    ]                   
    outcome = read_clinical_outcome(m_path, o_path, names)
    
    ## select wsi
    wsi_dir = pathlib.Path(args.wsi_dir)
    wsi_ids = list(outcome.keys())
    wsi_paths = sorted(wsi_dir.glob("*/*HE.isyntax"))
    wsi_paths = [p for p in wsi_paths if f"{p}".split("/")[-2] in wsi_ids]
    logging.info("Totally {} cases and {} slides!".format(len(wsi_ids), len(wsi_paths)))
    
    ## set save dir
    save_msk_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_masks")
    save_feature_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_features/{args.feature_mode}")
    save_model_dir = pathlib.Path(f"a_06semantic_segmentation/tile_{args.task}_models/{args.feature_mode}")
    save_result_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_results/{args.feature_mode}")

    ## generate wsi tissue mask
    # if args.mode == "wsi":
    #     generate_wsi_tissue_mask(
    #         wsi_paths=wsi_paths,
    #         save_msk_dir=save_msk_dir,
    #         method=args.mask_method,
    #         resolution=16*args.resolution,
    #         units=args.units
    #     )

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

    ## classify carcinoma
    # fold = 0
    # name = args.loss
    # scaler_path = f"{save_model_dir}/node_scaler.dat"
    # pretrained_main_classification = f"{save_model_dir}/BayesGIN_{name}/{fold:02d}/epoch=049.weights.pth"
    # pretrained_aux_classification = f"{save_model_dir}/BayesGIN_{name}/{fold:02d}/epoch=049.aux.dat"
    # pretrained_classification = [pretrained_main_classification, pretrained_aux_classification]
    # if name in ["PCE", "LHCE"]:
    #     pretrained_main_detection = f"{save_model_dir}/BayesGIN_nnPU/{fold:02d}/epoch=049.weights.pth"
    #     pretrained_aux_detection = f"{save_model_dir}/BayesGIN_nnPU/{fold:02d}/epoch=049.aux.dat"
    #     pretrained_detection = [pretrained_main_detection, pretrained_aux_detection]
    # else:
    #     pretrained_detection = None

    # mkdir(save_result_dir)
    # for idx, wsi_path in enumerate(wsi_paths):
    #     logging.info(f"Infering on wsi {idx + 1}/{len(wsi_paths)}...")
    #     wsi_name = pathlib.Path(wsi_path).stem 
    #     graph_path = save_feature_dir / f"{wsi_name}.json"
    #     prob = test(
    #         graph_path=graph_path,
    #         scaler_path=scaler_path,
    #         num_node_features=args.node_features,
    #         pretrained_classificaiton=pretrained_classification,
    #         pretrained_detection=pretrained_detection,
    #         conv="GINConv",
    #         BayesGNN=args.Bayes
    #     )
    #     save_pred_path = f"{save_result_dir}/{wsi_name}.pred.npy"
    #     logging.info(f"Saving predicted probability {wsi_name}.pred.npy")
    #     np.save(save_pred_path, prob)

    ## compute carcinoma size per-case
    case_ratios = compute_carcinoma_ratio(save_result_dir, wsi_ids, threshold=0.2)

    # survival, ratio = [], []
    # for k, v in outcome.items():
    #     if v[names[2]] == "Yes" and v[names[3]] == "Yes" and v[names[4]] == "Yes":
    #         survival.append(v[names[1]])
    #         ratio.append(case_ratios[k])
    # num_samples = len(survival)
    # if num_samples == 0:
    #     logging.info("No selected case")
    # else:
    #     survival = np.array(survival).mean()
    #     logging.info(f"Totally {num_samples} selected samples with mean survival length as {survival}")
    #     ratio = np.array(ratio).mean(axis=0)
    #     print("Mean carcinoma ratio is", ratio)

    # visualize scatter
    i = 2
    x1, x2, x3, x4, x5 = [], [], [], [], []
    y1, y2, y3, y4, y5 = [], [], [], [], []
    for k, v in outcome.items():
        if v[names[2]] == "Yes" and v[names[3]] == "No" and v[names[4]] == "No":
            x1.append(case_ratios[k][i])
            y1.append(v[names[1]])
        elif v[names[2]] == "No" and v[names[3]] == "Yes" and v[names[4]] == "No":
            x2.append(case_ratios[k][i])
            y2.append(v[names[1]])
        elif v[names[2]] == "Yes" and v[names[3]] == "Yes" and v[names[4]] == "No":
            x3.append(case_ratios[k][i])
            y3.append(v[names[1]])
        elif v[names[2]] == "No" and v[names[3]] == "Yes" and v[names[4]] == "Yes":
            x4.append(case_ratios[k][i])
            y4.append(v[names[1]])
        else:
            x5.append(case_ratios[k][i])
            y5.append(v[names[1]])
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    x3, y3 = np.array(x3), np.array(y3)
    x4, y4 = np.array(x4), np.array(y4)
    x5, y5 = np.array(x5), np.array(y5)
    a1, b1 = np.polyfit(x1, y1, 1)
    a2, b2 = np.polyfit(x2, y2, 1)
    a3, b3 = np.polyfit(x3, y3, 1)
    a4, b4 = np.polyfit(x4, y4, 1)
    plt.figure()
    plt.scatter(x1, y1, color="blue", marker="o", label="NMIBC")
    plt.plot(x1, a1 * x1 + b1, color="blue")
    plt.scatter(x2, y2, color="green", marker="^", label="MIBC")
    plt.plot(x2, a2 * x2 + b2, color="green")
    plt.scatter(x3, y3, color="orange", marker="+", label="NMIBC + MIBC")
    plt.plot(x3, a3 * x3 + b3, color="orange")
    plt.scatter(x4, y4, color="purple", marker="*", label="MIBC + Adjuvant")
    plt.plot(x4, a4 * x4 + b4, color="purple")
    plt.scatter(x5, y5, color="red", marker="x", label="others")
    plt.xlabel("Ratio of high-grade carcinoma")
    plt.ylabel("Length of time to death")
    plt.legend()
    plt.savefig("ratio_to_length.png")

        
