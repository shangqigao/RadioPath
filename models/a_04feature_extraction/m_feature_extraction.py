import sys
sys.path.append('../')

import random
import torch
import os
import pathlib
import joblib
import argparse
import pathlib
import timm
import cv2
import logging
import json
import PIL
import skimage

import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from common.m_utils import recur_find_ext, rmdir, select_checkpoints, mkdir
from tiatoolbox.models import DeepFeatureExtractor, IOSegmentorConfig, NucleusInstanceSegmentor
from tiatoolbox.models.architecture.vanilla import CNNBackbone, CNNModel
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.data import stain_norm_target
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils.misc import imwrite

from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree
from pprint import pprint

from radiomics import featureextractor
import radiomics

from monai.transforms.utils import generate_spatial_bounding_box
from monai.transforms.utils import get_largest_connected_component_mask

SEED = 5
random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def extract_pathomic_feature(
        wsi_paths, 
        wsi_msk_paths, 
        feature_mode, 
        save_dir, 
        mode, 
        resolution=0.5, 
        units="mpp"
    ):
    """extract pathomic feature from wsi
    Args:
        wsi_paths (list): a list of wsi paths
        wsi_msk_paths (list): a list of tissue mask paths of wsi
        fature_mode (str): mode of extracting features, 
            "composition" for extracting features by segmenting and counting nucleus
            "cnn" for extracting features by deep neural networks
        save_dir (str): directory of saving features
        mode (str): 'wsi' or 'tile', if 'wsi', extracting features of wsi
            could be slow if feature mode if 'composition'
        resolution (int): the resolution of extacting features
        units (str): the units of resolution, e.g., mpp  
    """
    if feature_mode == "cnn":
        _ = extract_cnn_pathomic_features(
            wsi_paths,
            wsi_msk_paths,
            save_dir,
            mode,
            resolution,
            units
        )
    elif feature_mode == "vit":
        _ = extract_vit_pathomic_features(
            wsi_paths,
            wsi_msk_paths,
            save_dir,
            mode,
            resolution,
            units
        )
    elif feature_mode == "uni":
        _ = extract_uni_pathomic_features(
            wsi_paths,
            wsi_msk_paths,
            save_dir,
            mode,
            resolution,
            units
        )
    elif feature_mode == "conch":
        _ = extract_conch_pathomic_features(
            wsi_paths,
            wsi_msk_paths,
            save_dir,
            mode,
            resolution,
            units
        )
    elif feature_mode == "chief":
        _ = extract_chief_pathomic_features(
            wsi_paths,
            wsi_msk_paths,
            save_dir,
            mode,
            resolution,
            units
        )
    else:
        raise NotImplementedError
    return

def extract_radiomic_feature(
        img_paths, 
        lab_paths, 
        feature_mode, 
        save_dir, 
        class_name,
        label=1,
        resolution=None, 
        units="mm",
        n_jobs=32
    ):
    """extract pathomic feature from wsi
    Args:
        img_paths (list): a list of image paths
        lab_paths (list): a list of label paths
        fature_mode (str): mode of extracting features, 
            "pyradiomics" for extracting radiomics
        save_dir (str): directory of saving features
        label (int): value for which to extract features
        resolution (int): the resolution of extacting features
        units (str): the units of resolution, e.g., mpp  

    """
    if feature_mode == "pyradiomics":
        _ = extract_pyradiomics(
            img_paths,
            lab_paths,
            save_dir,
            class_name,
            label,
            resolution,
            units,
            n_jobs
        )
    elif feature_mode == "SegVol":
        _ = extract_ViTradiomics(
            img_paths,
            lab_paths,
            save_dir,
            class_name,
            label,
            resolution,
            units
        )
    elif feature_mode == "M3D-CLIP":
        _ = extract_M3DCLIPradiomics(
            img_paths,
            lab_paths,
            save_dir,
            class_name,
            label,
            resolution,
            units
        )
    else:
        raise ValueError(f"Invalid feature mode: {feature_mode}")
    return

def extract_cnn_pathomic_features(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp"):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[224, 224],
        patch_output_shape=[224, 224],
        stride_shape=[224, 224],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    model = CNNBackbone("resnet50")
    ## define preprocessing function
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    TS = A.Compose([A.Normalize(mean, std), ToTensorV2()])
    def _preproc_func(img):
        return TS(image=img)["image"]
    def _postproc_func(img):
        return img
    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func
    
    extractor = DeepFeatureExtractor(
        batch_size=128, 
        model=model, 
        num_loader_workers=32, 
    )

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = extractor.predict(
        wsi_paths,
        msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=tmp_save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent.parent

        src_path = pathlib.Path(f"{output_path}.position.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.position.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}.features.0.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.features.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

class CNNClassifier(CNNModel):
    def __init__(self, backbone, num_classes=1):
        super().__init__(backbone, num_classes)
    
    def forward(self, imgs):
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        return torch.flatten(gap_feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        image = batch_data.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(image)
        return [output.cpu().numpy()]    

    def load(self, feature_path, classifier_path):
        feature_state_dict = torch.load(feature_path)
        self.feat_extract.load_state_dict(feature_state_dict)
        classifier_state_dict = torch.load(classifier_path)
        self.classifier.load_state_dict(classifier_state_dict)

class ViT(torch.nn.Module):
    def __init__(self, model256_path):
        super().__init__()
        from tiatoolbox.models.architecture.hipt import get_vit256
        self.model256 = get_vit256(pretrained_weights=model256_path)
    
    def forward(self, imgs):
        feat = self.model256(imgs)
        return torch.flatten(feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        image = batch_data.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(image)
        return [output.cpu().numpy()]
    
def extract_vit_pathomic_features(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp"):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[256, 256],
        patch_output_shape=[256, 256],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    pretrained_path = "../checkpoints/HIPT/vit256_small_dino.pth"
    model = ViT(pretrained_path)
    ## define preprocessing function
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    TS = A.Compose([A.Normalize(mean, std), ToTensorV2()])
    def _preproc_func(img):
        return TS(image=img)["image"]
    def _postproc_func(img):
        return img
    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func

    extractor = DeepFeatureExtractor(
        batch_size=128, 
        model=model, 
        num_loader_workers=32, 
    )

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = extractor.predict(
        wsi_paths,
        msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=tmp_save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent.parent

        src_path = pathlib.Path(f"{output_path}.position.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.position.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}.features.0.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.features.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

class UNI(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = timm.create_model(
            model_name="vit_large_patch16_224", 
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True,
            checkpoint_path=model_path
        )
    
    def forward(self, imgs):
        feat = self.model(imgs)
        return torch.flatten(feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        image = batch_data.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(image)
        return [output.cpu().numpy()]
    
def extract_uni_pathomic_features(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp"):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[256, 256],
        patch_output_shape=[256, 256],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    pretrained_path = "../checkpoints/UNI/pytorch_model.bin"
    model = UNI(pretrained_path)
    ## define preprocessing function
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    TS = A.Compose([A.Resize(224, 224, cv2.INTER_CUBIC), A.Normalize(mean, std), ToTensorV2()])
    def _preproc_func(img):
        return TS(image=img)["image"]
    def _postproc_func(img):
        return img
    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func

    extractor = DeepFeatureExtractor(
        batch_size=128, 
        model=model, 
        num_loader_workers=32, 
    )

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = extractor.predict(
        wsi_paths,
        msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=tmp_save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent.parent

        src_path = pathlib.Path(f"{output_path}.position.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.position.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}.features.0.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.features.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

class CONCH(torch.nn.Module):
    def __init__(self, cfg, ckpt_path, device):
        super().__init__()
        from tiatoolbox.models.architecture.conch.open_clip_custom import create_model_from_pretrained
        self.model, self.preprocess = create_model_from_pretrained(cfg, ckpt_path, device)

    def forward(self, images):
        img_embeddings = self.model.encode_image(images, proj_contrast=False, normalize=False)
        return torch.flatten(img_embeddings, 1)

    @staticmethod
    def infer_batch(model, images, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        images = images.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(images)
        return [output.cpu().numpy()]
    
def extract_conch_pathomic_features(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp"):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[256, 256],
        patch_output_shape=[256, 256],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    model_cfg = 'conch_ViT-B-16'
    ckpt_path = '../checkpoints/CONCH/pytorch_model.bin'
    model = CONCH(model_cfg, ckpt_path, 'cuda')
    ## define preprocessing function
    TS = model.preprocess
    def _preproc_func(img):
        img = PIL.Image.fromarray(img)
        return TS(img)
    def _postproc_func(img):
        return img
    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func

    extractor = DeepFeatureExtractor(
        batch_size=128, 
        model=model, 
        num_loader_workers=32, 
    )

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = extractor.predict(
        wsi_paths,
        msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=tmp_save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent.parent

        src_path = pathlib.Path(f"{output_path}.position.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.position.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}.features.0.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.features.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

class CHIEF(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        from tiatoolbox.models.architecture.chief.ctran import ConvStem
        from tiatoolbox.models.architecture.chief.timm.timm import create_model
        self.model = create_model(
            'swin_tiny_patch4_window7_224', 
            embed_layer=ConvStem, 
            pretrained=False
        )
        self.model.head = torch.nn.Identity()
        td = torch.load(model_path)
        self.model.load_state_dict(td['model'], strict=True)
    
    def forward(self, imgs):
        feat = self.model(imgs)
        return torch.flatten(feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        image = batch_data.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(image)
        return [output.cpu().numpy()]

def extract_chief_pathomic_features(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp"):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[256, 256],
        patch_output_shape=[256, 256],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    pretrained_path = "../checkpoints/CHIEF/CHIEF_CTransPath.pth"
    model = CHIEF(pretrained_path)
    ## define preprocessing function
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    TS = A.Compose([A.Resize(224, 224, cv2.INTER_CUBIC), A.Normalize(mean, std), ToTensorV2()])
    def _preproc_func(img):
        return TS(image=img)["image"]
    def _postproc_func(img):
        return img
    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func

    extractor = DeepFeatureExtractor(
        batch_size=128, 
        model=model, 
        num_loader_workers=32, 
    )

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = extractor.predict(
        wsi_paths,
        msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=tmp_save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent.parent

        src_path = pathlib.Path(f"{output_path}.position.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.position.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}.features.0.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.features.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

def extract_chief_wsi_level_features(patch_feature_paths, anatomic=13, on_gpu=True):
    from tiatoolbox.models.architecture.chief.CHIEF import CHIEF
    text_embedding_path = '../checkpoints/CHIEF/Text_emdding.pth'
    model = CHIEF(size_arg="small", dropout=True, n_classes=2, text_embedding_path=text_embedding_path)
    td = torch.load('../checkpoints/CHIEF/CHIEF_pretraining.pth')
    model.load_state_dict(td, strict=True)
    device = "cuda" if on_gpu else "cpu"
    model.to(device)
    model.eval()

    for i, path in enumerate(patch_feature_paths):
        features = np.load(path)
        with torch.no_grad():
            x = torch.tensor(features).to(device)
            anatomic = torch.tensor([anatomic]).to(device)
            result = model(x, anatomic)
            wsi_feature_emb = result['WSI_feature'].squeeze().cpu().numpy()
        save_path = f"{path}".replace(".features.npy", ".WSI.features.npy")
        save_name = save_path.split("/")[-1]
        logging.info(f"Saving [{i+1}/{len(patch_feature_paths)}] WSI-level features as {save_name} ...")
        np.save(save_path, wsi_feature_emb)
    return

def extract_composition_features(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp"):
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=16,
        num_loader_workers=8,
        num_postproc_workers=8,
    )
    if mode == "wsi":
        inst_segmentor.ioconfig.tile_shape = (5120, 5120)
    
    ## define preprocessing function
    target_image = stain_norm_target()
    stain_normaliser = get_normalizer("reinhard")
    stain_normaliser.fit(target_image)
    def _stain_norm_func(img):
        return stain_normaliser.transform(img)
    inst_segmentor.model.preproc_func = _stain_norm_func

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = inst_segmentor.predict(
        wsi_paths,
        msk_paths,
        mode=mode,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir,
        resolution=resolution,
        units=units,
    )

    output_paths = []
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent

        src_path = pathlib.Path(f"{output_path}.dat")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.dat")
        src_path.rename(new_path)
        output_paths.append(new_path)
    for idx, path in enumerate(output_paths):
        if msk_paths is not None:
            get_cell_compositions(wsi_paths[idx], msk_paths[idx], path, save_dir, resolution=resolution, units=units)
        else:
            get_cell_compositions(wsi_paths[idx], None, path, save_dir, resolution=resolution, units=units)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_paths


def get_cell_compositions(
        wsi_path,
        mask_path,
        inst_pred_path,
        save_dir,
        num_types = 2,
        patch_input_shape = (512, 512),
        stride_shape = (512, 512),
        resolution = 0.5,
        units = "mpp",
):
    if pathlib.Path(wsi_path).suffix == ".jpg":
        reader = WSIReader.open(wsi_path, mpp=(resolution, resolution))
    else:
        reader = WSIReader.open(wsi_path)
    inst_pred = joblib.load(inst_pred_path)
    inst_pred = {i: v for i, (_, v) in enumerate(inst_pred.items())}
    inst_boxes = [v["box"] for v in inst_pred.values()]
    inst_boxes = np.array(inst_boxes)

    geometries = [shapely_box(*bounds) for bounds in inst_boxes]
    spatial_indexer = STRtree(geometries)
    wsi_shape = reader.slide_dimensions(resolution=resolution, units=units)

    (patch_inputs, _) = PatchExtractor.get_coordinates(
        image_shape=wsi_shape,
        patch_input_shape=patch_input_shape,
        patch_output_shape=patch_input_shape,
        stride_shape=stride_shape,
    )

    if mask_path is not None:
        mask_reader = WSIReader.open(mask_path)
        mask_reader.info = reader.info
        selected_coord_indices = PatchExtractor.filter_coordinates(
            mask_reader,
            patch_inputs,
            wsi_shape=wsi_shape,
            min_mask_ratio=0.5,
        )
        patch_inputs = patch_inputs[selected_coord_indices]

    bounds_compositions = []
    for bounds in patch_inputs:
        bounds_ = shapely_box(*bounds)
        indices = [geo for geo in spatial_indexer.query(bounds_) if bounds_.contains(geometries[geo])]
        insts = [inst_pred[v]["type"] for v in indices]
        _, freqs = np.unique(insts, return_counts=True)
        holder = np.zeros(num_types, dtype=np.int16)
        holder[0] = freqs.sum()
        bounds_compositions.append(holder)
    bounds_compositions = np.array(bounds_compositions)

    base_name = pathlib.Path(wsi_path).stem
    np.save(f"{save_dir}/{base_name}.position.npy", patch_inputs)
    np.save(f"{save_dir}/{base_name}.features.npy", bounds_compositions)

def extract_pyradiomics(img_paths, lab_paths, save_dir, class_name, label=None, resolution=None, units="mm", n_jobs=32):
    # Get the PyRadiomics logger (default log-level = INFO)
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

    # Write out all log entries to a file
    handler = logging.FileHandler(filename=f"testLog.{class_name}.txt", mode='w')
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    settings = {}
    settings['resampledPixelSpacing'] = [resolution, resolution, resolution]
    settings['correctMask'] = True

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableImageTypeByName('Wavelet')
    os.makedirs(save_dir, exist_ok=True)
    def _extract_radiomics(idx, img_path, lab_path):
        logging.info("extracting radiomics: {}/{}...".format(idx + 1, len(img_paths)))
        features = extractor.execute(img_path, lab_path, label)
        for k, v in features.items():
            if isinstance(v, np.ndarray):
                features[k] = v.tolist()
        img_name = f"{img_path}".split("/")[-1].replace(".nii.gz", "")
        save_path = pathlib.Path(f"{save_dir}/{img_name}.{class_name}.pyradiomics.json")
        with save_path.open("w") as handle:
            json.dump(features, handle, indent=4)
        return

    # extract radiomics in parallel
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_extract_radiomics)(idx, img_path, lab_path)
        for idx, (img_path, lab_path) in enumerate(zip(img_paths, lab_paths))
    )
    return

def extract_VOI(image, label, patch_size, padding, output_shape=None):
    assert image.ndim == 3
    label = get_largest_connected_component_mask(label)
    s, e = generate_spatial_bounding_box(np.expand_dims(label, 0))
    s = np.array(s) - np.array(padding)
    e = np.array(e) + np.array(padding)
    image = image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
    if output_shape is not None:
        shape = output_shape
    else:
        shape = image.shape * np.array(patch_size, np.int32)
    image = skimage.transform.resize(image, output_shape=shape)
    bbox = [s, e]
    return image, bbox

def SegVol_image_transforms(keys, spacing, padding):
    from monai import transforms

    class MinMaxNormalization(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys

        def __call__(self, data):
            d = dict(data)
            for k in self.keys:
                d[k] = d[k] - d[k].min()
                d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
            return d

    class ForegroundNormalization(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys
        
        def __call__(self, data):
            d = dict(data)
            
            for key in self.keys:
                d[key] = self.normalize(d[key])
            return d
        
        def normalize(self, ct_narray):
            ct_voxel_ndarray = ct_narray.copy()
            ct_voxel_ndarray = ct_voxel_ndarray.flatten()
            thred = np.mean(ct_voxel_ndarray)
            voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
            upper_bound = np.percentile(voxel_filtered, 99.95)
            lower_bound = np.percentile(voxel_filtered, 00.05)
            mean = np.mean(voxel_filtered)
            std = np.std(voxel_filtered)
            ### transform ###
            ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
            ct_narray = (ct_narray - mean) / max(std, 1e-8)
            return ct_narray

    class DimTranspose(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys
        
        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                d[key] = np.swapaxes(d[key], -1, -3)
            return d

    transform = transforms.Compose(
            [
                transforms.LoadImaged(keys, ensure_channel_first=True, allow_missing_keys=True),
                transforms.Spacingd(keys, pixdim=spacing, mode=('bilinear', 'nearest')),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                ForegroundNormalization(keys=["image"]),
                MinMaxNormalization(keys=["image"]),
                DimTranspose(keys=["image", "label"]),
                transforms.BorderPadd(keys=["image", "label"], spatial_border=padding)
            ]
        )
    return transform


def extract_ViTradiomics(img_paths, lab_paths, save_dir, class_name, label=1, resolution=1.024, units="mm", device="cuda"):
    from monai.networks.nets import ViT
    from monai.inferers import SlidingWindowInferer
    
    roi_size = (32, 256, 256)
    patch_size = (4, 16, 16)
    vit = ViT(
        in_channels=1,
        img_size=roi_size,
        patch_size=patch_size,
        pos_embed="perceptron",
        )
    print(vit)
    vit_checkpoint = '../checkpoints/SegVol/ViT_pretrain.ckpt'
    with open(vit_checkpoint, "rb") as f:
        state_dict = torch.load(f, map_location='cpu')['state_dict']
        encoder_dict = {k.replace('model.encoder.', ''): v for k, v in state_dict.items() if 'model.encoder.' in k}
    vit.load_state_dict(encoder_dict)
    vit.to(device)
    vit.eval()
    print(f'Loaded SegVol encoder param: {vit_checkpoint}')

    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=8,
        sw_device=device,
        device='cpu',
        progress=True
    )
    print("Set sliding window for model inference.")

    spacing = (resolution, resolution, resolution)
    keys = ["image", "label"]
    padding = (4, 8, 8)
    transform = SegVol_image_transforms(keys, spacing, padding)
    case_dicts = [
        {"image": img_path, "label": lab_path} for img_path, lab_path in zip(img_paths, lab_paths)
    ]
    data_dicts = transform(case_dicts)
    fs = (np.array(roi_size) / np.array(patch_size)).astype(np.int32)
    mkdir(save_dir)
    
    for case, data in zip(case_dicts, data_dicts):
        image = data["image"].squeeze().numpy()
        label = data["label"].squeeze().numpy()
        voi, bbox = extract_VOI(image, label, patch_size, padding)
        img_shape, voi_shape = image.shape, voi.shape
        voi = torch.from_numpy(voi).unsqueeze(0).unsqueeze(0).to('cpu')
        with torch.no_grad():
            feature = inferer(voi, lambda x: vit(x)[0].transpose(1, 2).reshape(-1, 768, fs[0], fs[1], fs[2]))
        c, z, x, y = feature.squeeze().size()
        feat_shape = (c, z, x, y)
        logging.info(f"Got image of shape {img_shape}, VOI of shape {voi_shape}, feature of shape {feat_shape}")
        feature = feature.squeeze().reshape([c, z*x*y]).transpose(0,1).cpu().numpy()
        z, x, y = np.arange(z), np.arange(x), np.arange(y)
        Z, X, Y = np.meshgrid(z, x, y, indexing="ij")
        coordinates = np.array([bbox[0]]) + np.stack([Z, X, Y], axis=-1).reshape([-1, 3])
        assert len(feature) == len(coordinates)
        logging.info(f"Saving radiomics in the resolution of {spacing}...")
        img_name = pathlib.Path(case["image"]).name.replace(".nii.gz", "")
        feature_path = f"{save_dir}/{img_name}_{class_name}_radiomics.npy"
        np.save(feature_path, feature)
        coordinates_path = f"{save_dir}/{img_name}_{class_name}_coordinates.npy"
        np.save(coordinates_path, coordinates)
    return

def M3DCLIP_image_transforms(keys, padding):
    from monai import transforms

    class MinMaxNormalization(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys

        def __call__(self, data):
            d = dict(data)
            for k in self.keys:
                d[k] = d[k] - d[k].min()
                d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
            return d

    class ForegroundNormalization(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys
        
        def __call__(self, data):
            d = dict(data)
            
            for key in self.keys:
                d[key] = self.normalize(d[key])
            return d
        
        def normalize(self, ct_narray):
            ct_voxel_ndarray = ct_narray.copy()
            ct_voxel_ndarray = ct_voxel_ndarray.flatten()
            thred = np.mean(ct_voxel_ndarray)
            voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
            upper_bound = np.percentile(voxel_filtered, 99.95)
            lower_bound = np.percentile(voxel_filtered, 00.05)
            mean = np.mean(voxel_filtered)
            std = np.std(voxel_filtered)
            ### transform ###
            ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
            ct_narray = (ct_narray - mean) / max(std, 1e-8)
            return ct_narray

    class DimTranspose(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys
        
        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                d[key] = np.swapaxes(d[key], -1, -3)
            return d

    transform = transforms.Compose(
            [
                transforms.LoadImaged(keys, ensure_channel_first=True, allow_missing_keys=True),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                ForegroundNormalization(keys=["image"]),
                MinMaxNormalization(keys=["image"]),
                DimTranspose(keys=["image", "label"]),
                transforms.BorderPadd(keys=["image", "label"], spatial_border=padding)
            ]
        )
    return transform

def extract_M3DCLIPradiomics(img_paths, lab_paths, save_dir, class_name, label=1, resolution=1.024, units="mm", device="cpu"):
    from transformers import AutoTokenizer, AutoModel
    
    roi_size = (32, 256, 256)
    tokenizer = AutoTokenizer.from_pretrained(
        "GoodBaiBai88/M3D-CLIP",
        model_max_length=512,
        padding_side="right",
        use_fast=False
    )
    model = AutoModel.from_pretrained(
        "GoodBaiBai88/M3D-CLIP",
        trust_remote_code=True
    )
    model = model.to(device=device)

    keys = ["image", "label"]
    padding = (4, 8, 8)
    transform = M3DCLIP_image_transforms(keys, padding)
    case_dicts = [
        {"image": img_path, "label": lab_path} for img_path, lab_path in zip(img_paths, lab_paths)
    ]
    data_dicts = transform(case_dicts)
    mkdir(save_dir)
    
    for case, data in zip(case_dicts, data_dicts):
        image = data["image"].squeeze().numpy()
        label = data["label"].squeeze().numpy()
        voi, bbox = extract_VOI(image, label, None, padding, roi_size)
        img_shape, voi_shape = image.shape, voi.shape
        voi = torch.from_numpy(voi).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            feature = model.encode_image(voi)[:, 0]
        feat_shape = feature.shape
        logging.info(f"Got image of shape {img_shape}, VOI of shape {voi_shape}, feature of shape {feat_shape}")
        feature = feature.squeeze().cpu().numpy()
        logging.info(f"Saving radiomics...")
        img_name = pathlib.Path(case["image"]).name.replace(".nii.gz", "")
        feature_path = f"{save_dir}/{img_name}_{class_name}_radiomics.npy"
        np.save(feature_path, feature)
        coordinates_path = f"{save_dir}/{img_name}_{class_name}_coordinates.npy"
        np.save(coordinates_path, np.array(bbox))
    return



if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/3923_21/3923_21_G_HE.isyntax")
    parser.add_argument('--mask_method', default='otsu', help='method of tissue masking')
    parser.add_argument('--tile_location', default=[50000, 50000], type=list)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--save_dir', default="a_04feature_extraction/wsi_features", type=str)
    parser.add_argument('--mode', default="tile", type=str)
    parser.add_argument('--feature_mode', default="cnn", type=str)
    args = parser.parse_args()

    wsi = WSIReader.open(args.slide_path)
    mask = wsi.tissue_mask(method=args.mask_method, resolution=1.25, units="power")
    pprint(wsi.info.as_dict())
    if args.mode == "tile":
        tile = wsi.read_region(args.tile_location, args.level, args.tile_size)
        wsi_path = os.path.join("a_04feature_extraction", 'tile_sample.jpg')
        imwrite(wsi_path, tile)
        tile_mask = mask.read_region(args.tile_location, args.level, args.tile_size)
        msk_path = os.path.join("a_04feature_extraction", 'tile_mask.jpg')
        imwrite(msk_path, np.uint8(tile_mask*255))
    elif args.mode == "wsi":
        wsi_mask = mask.slide_thumbnail(resolution=1.25, units="power")
        msk_path = os.path.join("a_04feature_extraction", 'wsi_mask.jpg')
        imwrite(msk_path, wsi_mask)
        wsi_path = args.slide_path
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    wsi_feature_dir = os.path.join(args.save_dir, args.feature_mode)
    if args.feature_mode == "composition":
        output_list = extract_composition_features(
            [wsi_path],
            [msk_path],
            wsi_feature_dir,
            args.mode,
        )
    elif args.feature_mode == "cnn":
        output_list = extract_cnn_pathomic_features(
            [wsi_path],
            [msk_path],
            wsi_feature_dir,
            args.mode,
        )
    else:
        raise ValueError(f"Invalid feature mode: {args.feature_mode}")






