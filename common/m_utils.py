import os
import pathlib
import logging
import json
import joblib
import shutil
from tqdm import tqdm
from pathlib import Path

from tiatoolbox.wsicore.wsireader import WSIReader 

def mkdir(dir_path: Path):
    """Create a directory if it does not exist."""
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)
    return

def rmdir(dir_path: str):
    """Remove a directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return

def rm_n_mkdir(dir_path: Path):
    """Remove then re-create a directory."""
    if dir_path.is_dir():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)
    return

def recur_find_ext(root_dir: Path, exts):
    """Recursively find files with an extension in `exts`.

    This is much faster than glob if the folder
    hierachy is complicated and contain > 1000 files.

    Args:
        root_dir (Path):
            Root directory for searching.
        exts (list):
            List of extensions to match.

    Returns:
        List of full paths with matched extension in sorted order.

    """
    assert isinstance(exts, list)
    file_path_list = []
    for cur_path, _dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = Path(file_name).suffix
            if file_ext in exts:
                full_path = f"{cur_path}/{file_name}"
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list

def load_json(path: str):
    path = pathlib.Path(path)
    with path.open() as fptr:
        return json.load(fptr)
    
def create_pbar(subset_name: str, num_steps: int):
    """Create a nice progress bar."""
    pbar_format = (
        "Processing: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    )
    pbar = tqdm(total=num_steps, leave=True, bar_format=pbar_format, ascii=True)
    if subset_name == "train":
        pbar_format += "step={postfix[1][step]:03d}|EMA={postfix[1][EMA]:0.5f}"
        # * Changing print char may break the bar so avoid it
        pbar = tqdm(
            total=num_steps,
            leave=True,
            initial=0,
            bar_format=pbar_format,
            ascii=True,
            postfix=["", {"step": int(999), "EMA": float("NaN")}],
        )
    return pbar

def reset_logging(save_path):
    """Reset logger handler."""
    log_formatter = logging.Formatter(
        "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
    )
    log = logging.getLogger()  # Root logger
    for hdlr in log.handlers[:]:  # Remove all old handlers
        log.removeHandler(hdlr)
    new_hdlr_list = [
        logging.FileHandler(f"{save_path}/debug.log"),
        logging.StreamHandler(),
    ]
    for hdlr in new_hdlr_list:
        hdlr.setFormatter(log_formatter)
        log.addHandler(hdlr)

def select_checkpoints(
    stat_file_path: str,
    top_k: int = 2,
    metric: str = "infer-valid-auprc",
    epoch_range = None,
):
    """Select checkpoints basing on training statistics.

    Args:
        stat_file_path (str): Path pointing to the .json
            which contains the statistics.
        top_k (int): Number of top checkpoints to be selected.
        metric (str): The metric name saved within .json to perform
            selection.
        epoch_range (list): The range of epochs for checking, denoted
            as [start, end] . Epoch x that is `start <= x <= end` is
            kept for further selection.

    Returns:
        paths (list): List of paths or info tuple where each point
            to the correspond check point saving location.
        stats (list): List of corresponding statistics.

    """
    if epoch_range is None:
        epoch_range = [0, 1000]
    stats_dict = load_json(stat_file_path)
    # k is the epoch counter in this case
    stats_dict = {
        k: v
        for k, v in stats_dict.items()
        if int(k) >= epoch_range[0] and int(k) <= epoch_range[1]
    }
    stats = [[int(k), v[metric], v] for k, v in stats_dict.items()]
    # sort epoch ranking from largest to smallest
    # stats = sorted(stats, key=lambda v: v[1], reverse=True)
    stats = stats[::-1]
    chkpt_stats_list = stats[:top_k]  # select top_k

    model_dir = Path(stat_file_path).parent
    epochs = [v[0] for v in chkpt_stats_list]
    paths = [
        (
            f"{model_dir}/epoch={epoch:03d}.weights.pth",
            f"{model_dir}/epoch={epoch:03d}.aux.dat",
        )
        for epoch in epochs
    ]
    chkpt_stats_list = [[v[0], v[2]] for v in chkpt_stats_list]
    print(paths)  # noqa: T201
    return paths, chkpt_stats_list

def select_wsi(wsi_dir: str, excluded_wsi: list):
    """select annotated wsi
    Args:
        wsi_dir (str): directory of wsi
    Returns:
        selected_wsi_paths (list[pathlib.Path]): a list of selected wsi paths
    """
    
    wsi_paths = sorted(pathlib.Path(wsi_dir).rglob("*.svs"))
    logging.info(f"Totally {len(wsi_paths)} WSIs!")

    def _filter_wsi(wsi_path):
        wsi = WSIReader.open(wsi_path)
        if wsi.info.mpp is None and wsi.info.objective_power is None:
            print("No wsi info:", f"{wsi_path}".split("/")[-1].split(".")[0])
            selected_path = None
        else:
            selected_path = wsi_path
        del wsi
        return selected_path

    selected_wsi_paths = joblib.Parallel(n_jobs=8)(
        joblib.delayed(_filter_wsi)(wsi_path) 
        for wsi_path in wsi_paths
        )

    selected_paths = []
    for path in selected_wsi_paths:
        wsi_name = f"{path}".split("/")[-1].split(".")[0]
        if path is not None and wsi_name not in excluded_wsi:
            selected_paths.append(path)
    return sorted(selected_paths)

def select_wsi_annotated(wsi_dir: str, ann_dir: str):
    """select annotated wsi
    Args:
        wsi_dir (str): directory of wsi
        ann_dir (str): directory of annotation
    Returns:
        selected_wsi_paths (list[pathlib.Path]): a list of selected wsi paths
        selected_ann_paths (list[pathlib.Path]): a list of selected annotation paths
    """
    
    ann_paths = sorted(pathlib.Path(ann_dir).rglob("*.json"))
    wsi_paths = sorted(pathlib.Path(wsi_dir).rglob("*HE.isyntax"))
    def _match_ann_wsi(ann_path, wsi_path):
        ann_name = ann_path.stem
        wsi_name = wsi_path.stem
        selected_paths = None
        if ann_name == wsi_name:
            selected_paths = [ann_path, wsi_path]
        return selected_paths

    selected_ann_wsi_paths = joblib.Parallel(n_jobs=8)(
        joblib.delayed(_match_ann_wsi)(ann_path, wsi_path) 
        for ann_path in ann_paths 
        for wsi_path in wsi_paths
        )

    selected_ann_paths = []
    selected_wsi_paths = []
    for paths in selected_ann_wsi_paths:
        if paths is not None:
            selected_ann_paths.append(paths[0])
            selected_wsi_paths.append(paths[1])
    return sorted(selected_wsi_paths), sorted(selected_ann_paths) 

def select_wsi_interested(wsi_names, wsi_paths, wsi_ann_paths):
    selected_wsi_paths, selected_wsi_ann_paths = [], []
    for wsi_path, ann_path in zip(wsi_paths, wsi_ann_paths):
        wsi_name = pathlib.Path(wsi_path).stem
        ann_name = pathlib.Path(ann_path).stem
        if wsi_name in wsi_names and ann_name in wsi_names:
            selected_wsi_paths.append(wsi_path)
            selected_wsi_ann_paths.append(ann_path)
    return selected_wsi_paths, selected_wsi_ann_paths

def concat_dict_list(dict_list):
    concat_dict = dict_list[0]
    if len(dict_list) > 1:
        for d in dict_list[1:]:
            for k, v in d.items(): concat_dict[k] += v
    return concat_dict
