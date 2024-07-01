import sys
sys.path.append('../')

import cv2
import pathlib
import logging
import copy
import torch
import joblib
import argparse
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import accuracy_score as acc_scorer
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tiatoolbox.utils.misc import save_as_json
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox import logger
from tiatoolbox.models.architecture.vanilla import CNNModel

from common.m_utils import mkdir, create_pbar, load_json, rm_n_mkdir
from common.m_utils import reset_logging, select_wsi_annotated

from models.a_02tissue_masking.m_tissue_masking import generate_wsi_tissue_mask
from models.a_03patch_extraction.m_patch_extraction import prepare_annotation_reader
from models.a_05feature_aggregation.m_graph_neural_network import ScalarMovingAverage
from models.a_05feature_aggregation.m_graph_neural_network import PULoss

torch.multiprocessing.set_sharing_strategy("file_system")

import warnings
warnings.filterwarnings('ignore')
    
class SlideDataset(Dataset):
    """loading graph data from disk
    """
    def __init__(self, info_list, lab_dict, tissue_masker, mode="train", resolution=0.5, units="mpp"):
        super().__init__()
        self.info_list = info_list
        self.lab_dict = lab_dict
        self.tissue_masker = tissue_masker
        self.mode = mode
        self.resolution = resolution
        self.units = units
        self._shared_trans = self.shared_transforms()
        self._specific_trans = self.specific_transforms()
    
    def __getitem__(self, idx):
        if any(v in self.mode for v in ["train", "valid"]):
            wsi_path, ann_path = self.info_list[idx]
            wsi_reader, ann_reader, _ = prepare_annotation_reader(
                wsi_path=wsi_path,
                wsi_ann_path=ann_path,
                lab_dict=lab_dict,
                resolution=self.resolution,
                units=self.units
            )
            image = wsi_reader.slide_thumbnail(self.resolution, self.units)
            label = np.zeros(image.shape[0:2], np.uint8)
            for k, msk_reader in ann_reader.items():
                lab_value = lab_dict[k]
                mask = msk_reader.slide_thumbnail(self.resolution, self.units)
                label += lab_value * mask.astype(np.uint8)
            tissue_mask = self.tissue_masker.transform([image])[0]
            label = label * tissue_mask.astype(np.uint8)

            transformed = self._shared_trans(image=image, mask=label)
            image, label = transformed["image"], transformed["mask"]
            image = self._specific_trans(image=image)["image"]

            uids, freq = np.unique(label, return_counts=True)
            freq = freq[uids > 0]
            uids = uids[uids > 0]
            if len(uids) == 0:
                label = torch.tensor([0])
            else:
                ratio = freq / np.prod(label.shape)
                uids = uids[ratio > 0.1]
                freq = freq[ratio > 0.1]
                if len(uids) == 0:
                    label = torch.tensor([0])
                else:
                    index = np.argmax(freq)
                    label = torch.tensor([uids[index]])
            return image, label
        else:
            wsi_path = self.info_list[idx]
            assert self.units == "mpp", "units must be mpp"
            wsi_reader = WSIReader.open(wsi_path, mpp=self.resolution)
            wsi_reader = self.wsi_readers[idx]
            image = wsi_reader.slide_thumbnail(self.resolution, self.units)

            image = self._shared_trans(image=image)["image"]
            image = self._specific_trans(image=image)["image"]
            return image
    
    def __len__(self):
        return len(self.info_list)
    
    def shared_transforms(self):
        TS = A.Compose([
            A.RandomCrop(256, 256),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Rotate(45, cv2.INTER_NEAREST),
            A.CenterCrop(224, 224)
        ])
        return TS
    
    def specific_transforms(self):
        TS = A.Compose([
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
        return TS

def generate_data_split(
        x: list,
        y: list,
        lab_dict: dict,
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
        annotation = load_json(path)
        label = 0
        for k in annotation.keys():
            polygons = annotation[k]["points"]
            if len(polygons) > 0:
                for polygon in polygons:
                    if len(polygon) > 0:
                        label = lab_dict[k]
        l.append(label)
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

class CNNClassifier(CNNModel):
    def __init__(self, backbone, num_classes=1):
        super().__init__(backbone, num_classes)
        self.aux_model = {}

    def forward(self, imgs):
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        gap_feat = torch.flatten(gap_feat, 1)
        logit = self.classifier(gap_feat)
        return logit

    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer):
        image, label = batch_data
        device = "cuda" if on_gpu else "cpu"
        image = image.to(device).type(torch.float32)
        label = label.to(device)

        model.train()
        optimizer.zero_grad()
        pred = model(image)
        loss = loss(pred.squeeze(), label.squeeze())
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        label = label.cpu().numpy()
        return [loss, pred, label]

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        if len(batch_data) == 2:
            image, label = batch_data
            image = image.to(device).type(torch.float32)
            model.eval()
            with torch.inference_mode():
                pred = model(image)
            pred = pred.squeeze().cpu().numpy()
            label = label.squeeze().numpy()
            if pred.ndim == 1:
                label = np.array(label > 0, dtype=np.int32)
            return [pred, label]
        else:
            image = batch_data.to(device).type(torch.float32)
            model.eval()
            with torch.inference_mode():
                pred = model(image)
            pred = pred.squeeze().cpu().numpy()
            return [pred] 

    def save(self, extractor_path, classifier_path):
        feature_state_dict = self.feat_extract.state_dict()
        torch.save(feature_state_dict, extractor_path)
        classifier_state_dict = self.classifier.state_dict()
        torch.save(classifier_state_dict, classifier_path)

    def load(self, extractor_path, classifier_path):
        feature_state_dict = torch.load(extractor_path)
        self.feat_extract.load_state_dict(feature_state_dict)
        classifier_state_dict = torch.load(classifier_path)
        self.classifier.load_state_dict(classifier_state_dict)

def run_once(
        dataset_dict,
        num_epochs,
        save_dir,
        lab_dict,
        tissue_masker,
        on_gpu=True,
        pretrained=None,
        loader_kwargs=None,
        optim_kwargs=None,
        resolution=0.5,
        units="mpp",
):
    """running the inference or training loop once"""
    if loader_kwargs is None:
        loader_kwargs = {}

    if optim_kwargs is None:
        optim_kwargs = {}

    model = CNNClassifier("resnet50", num_classes=1)
    if pretrained is not None:
        model.load(*pretrained)
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
    loss = PULoss(prior=0.35, mode="nnPU")

    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        ds = SlideDataset(subset, lab_dict, tissue_masker, mode=subset_name, resolution=resolution, units=units)
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
                    output = model.train_batch(model, batch_data, on_gpu, loss, optimizer)
                    ema({"loss": output[0]})
                    pbar.postfix[1]["step"] = step
                    pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
                else:
                    output = model.infer_batch(model, batch_data, on_gpu)
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
                prob = prob[:, 1] if logit.shape[1] == 1 else prob
                val = auroc_scorer(true, prob, multi_class="ovr")
                logging_dict[f"{loader_name}-auroc"] = val

                if logit.shape[1] == 1:
                    val = auprc_scorer(true, prob)
                else:
                    onehot = np.eye(logit.shape[1])[true]
                    val = auprc_scorer(onehot, prob)
                logging_dict[f"{loader_name}-auprc"] = val

                logging_dict[f"{loader_name}-raw-logit"] = prob
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
                    f"{save_dir}/epoch={epoch:03d}.extractor.weights.pth",
                    f"{save_dir}/epoch={epoch:03d}.classifier.weights.pth",
                )
    
    return step_output


def training(
        num_epochs,
        split_path,
        model_dir,
        lab_dict,
        tissue_masker,
        resolution,
        units,
        pretrained=None,
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
    
    loader_kwargs = {
        "num_workers": 16, 
        "batch_size": 32,
    }
    optim_kwargs = {
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
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
            lab_dict=lab_dict,
            tissue_masker=tissue_masker,
            pretrained=pretrained,
            loader_kwargs=loader_kwargs,
            optim_kwargs=optim_kwargs,
            resolution=resolution,
            units=units
        )
    return

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', default="/well/rittscher/shared/datasets/KiBla/cases")
    parser.add_argument('--wsi_ann_dir', default="a_06semantic_segmentation/wsi_bladder_annotations")
    parser.add_argument('--save_dir', default="a_04feature_extraction", type=str)
    parser.add_argument('--mask_method', default='otsu', choices=["otsu", "morphological"], help='method of tissue masking')
    parser.add_argument('--task', default="bladder", choices=["bladder", "kidney"], type=str)
    parser.add_argument('--mode', default="tile", choices=["tile", "wsi"], type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--feature_mode', default="cnn", choices=["cnn", "composition"], type=str)
    parser.add_argument('--resolution', default=20, type=float)
    parser.add_argument('--units', default="power", type=str)
    args = parser.parse_args()

    ## select annotated wsi
    wsi_dir = pathlib.Path(args.wsi_dir)
    wsi_ann_dir = pathlib.Path(args.wsi_ann_dir)
    wsi_paths, wsi_ann_paths = select_wsi_annotated(wsi_dir, wsi_ann_dir)
    logging.info("Totally {} wsi and {} annotation!".format(len(wsi_paths), len(wsi_ann_paths)))
    
    ## set save dir
    save_tile_dir = pathlib.Path(f"a_06semantic_segmentation/wsi_{args.task}_tiles")
    save_model_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_models/{args.feature_mode}")
    

    ## define label name and value, should be consistent with annotation
    lab_dict = {
        "Background": 0,
        "Typical low grade urothelial carcinoma": 1,
        "Borderline/indeterminate for low grade vs high grade": 2,
        "Typical high grade urothelial carcinoma": 3,
    }
    ## set annotation resolution

    ## generate ROI tile from wsi based on annotation
    if args.mode == "tile":
        wsi_paths = sorted(save_tile_dir.glob("*.tile.jpg"))
        wsi_ann_paths = sorted(save_tile_dir.glob("*.annotation.json"))
        logging.info("Totally {} tile and {} annotation!".format(len(wsi_paths), len(wsi_ann_paths)))

    ## split data set
    num_folds = 5
    test_ratio = 0.2
    train_ratio = 0.8 * 0.9
    valid_ratio = 0.8 * 0.1
    splits = generate_data_split(
        x=wsi_paths,
        y=wsi_ann_paths,
        lab_dict=lab_dict,
        train=train_ratio,
        valid=valid_ratio,
        test=test_ratio,
        num_folds=num_folds,
    )
    num_train = len(splits[0]["train"])
    logging.info(f"Number of training samples: {num_train}.")
    num_valid = len(splits[0]["valid"])
    logging.info(f"Number of validating samples: {num_valid}.")
    num_test = len(splits[0]["test"])
    logging.info(f"Number of testing samples: {num_test}.")
    mkdir(save_model_dir)
    split_path = f"{save_model_dir}/splits.dat"
    joblib.dump(splits, split_path)

    ## training
    tissue_masker = generate_wsi_tissue_mask(
        wsi_paths=wsi_paths,
        method=args.mask_method,
        resolution=16*args.resolution,
        units="mpp"
    )
    training(
        num_epochs=args.epochs,
        split_path=split_path,
        model_dir=save_model_dir,
        lab_dict=lab_dict,
        tissue_masker=tissue_masker,
        resolution=args.resolution,
        units=args.units
    )




    



