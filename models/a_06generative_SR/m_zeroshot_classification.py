import sys
sys.path.append('../')

import torch
import os
import pathlib
import argparse
import pathlib
import json
import logging
import PIL

import numpy as np
import albumentations as A

from common.m_utils import rmdir
from albumentations.pytorch import ToTensorV2
from tiatoolbox.models import DeepFeatureExtractor, IOSegmentorConfig
from tiatoolbox.models.architecture.vanilla import CNNBackbone
from tiatoolbox.models.architecture.hipt import get_vit256
from tiatoolbox.models.architecture.conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import imwrite

from pprint import pprint

# SEED = 5
# random.seed(SEED)
# rng = np.random.default_rng(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

def pathology_zeroshot_classification(
        wsi_paths, 
        wsi_msk_paths, 
        cls_mode, 
        save_dir, 
        mode, 
        prompts,
        resolution=0.5, 
        units="mpp"
    ):
    """extract pathomic feature from wsi
    Args:
        wsi_paths (list): a list of wsi paths
        wsi_msk_paths (list): a list of tissue mask paths of wsi
        cls_mode (str): mode of zero-shot classification, 
            "MIzero" for zero-shot classifcation by MI-zero
            "CONCH" for zero-shot classfication by CONCH
        save_dir (str): directory of saving features
        mode (str): 'wsi' or 'tile', if 'wsi', extracting features of wsi
            could be slow if feature mode if 'composition'
        resolution (int): the resolution of extacting features
        units (str): the units of resolution, e.g., mpp  
    """
    if cls_mode == "mizero":
        _ = pathology_mizero_zeroshot_classification(
            wsi_paths,
            wsi_msk_paths,
            save_dir,
            mode,
            prompts,
            resolution,
            units
        )
    elif cls_mode == "conch":
        _ = pathology_conch_zeroshot_classification(
            wsi_paths,
            wsi_msk_paths,
            save_dir,
            mode,
            prompts,
            resolution,
            units
        )
    else:
        raise ValueError(f"Invalid feature mode: {cls_mode}")
    return

class MIzero(torch.nn.Module):
    def __init__(self, model256_path):
        super().__init__()
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

def pathology_mizero_zeroshot_classification(wsi_paths, msk_paths, save_dir, mode, prompts, resolution=0.5, units="mpp"):
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
        batch_size=32, 
        model=model, 
        num_loader_workers=8, 
    )

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
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
    def __init__(self, cfg, ckpt_path, prompts, device):
        super().__init__()
        self.model, self.preprocess = create_model_from_pretrained(cfg, ckpt_path, device)
        self.tokenizer = get_tokenizer()
        self.prompts = tokenize(self.tokenizer, prompts).to(device)

    def forward(self, images, prompts):
        img_embeddings = self.model.encode_image(images)
        txt_embeddings = self.model.encode_text(prompts)
        return img_embeddings, txt_embeddings

    @staticmethod
    def infer_batch(model, images, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        images = images.to(device).type(torch.float32)
        prompts = model.module.prompts # model is a DataParallel object
        model.eval()
        with torch.inference_mode():
            img_embedings, txt_embedings = model(images, prompts)
            sim_scores = (img_embedings @ txt_embedings.T * model.module.model.logit_scale.exp()).softmax(dim=-1)
        return [sim_scores.cpu().numpy()]
    
def pathology_conch_zeroshot_classification(wsi_paths, msk_paths, save_dir, mode, prompts, resolution=0.5, units="mpp"):
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
    model = CONCH(model_cfg, ckpt_path, prompts, 'cuda')
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
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.SimilarityScores.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

def load_prompts(json_path, index=10):
    with open(json_path) as f:
        prompts = json.load(f)['0']
    classnames = prompts['classnames']
    template = prompts['templates'][index]
    classnames_text = []
    for v in classnames.values():
        classnames_text += v
    logging.info(f"The number prompt classes: {len(classnames_text)}")
    if index > 0:
        prompts = [template.replace("CLASSNAME", name) for name in classnames_text]
    else:
        prompts = classnames_text
    return prompts



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
    parser.add_argument('--cls_mode', default="cnn", type=str)
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
    if args.cls_mode == "MIzero":
        output_list = pathology_mizero_zeroshot_classification(
            [wsi_path],
            [msk_path],
            wsi_feature_dir,
            args.mode,
        )
    elif args.cls_mode == "CONCH":
        output_list = pathology_conch_zeroshot_classification(
            [wsi_path],
            [msk_path],
            wsi_feature_dir,
            args.mode,
        )
    else:
        raise ValueError(f"Invalid feature mode: {args.feature_mode}")






