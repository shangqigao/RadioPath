import csv
import json
import os
import sys
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)

def mkdir(dir_path: Path):
    """Create a directory if it does not exist."""
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)
    return

def csv_to_json(csv_path, save_json_dir):
    save_json_dir = pathlib.Path(save_json_dir)
    mkdir(save_json_dir)
    with open(csv_path) as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            img_name = row["Image"]
            save_json_path = save_json_dir / f"{img_name}.json"
            dict_ann = json.loads(row["Annotation"])
            layers = dict_ann["annotation"]["layers"]
            dict_points = {}
            num_points = 0
            for layer in layers:
                name = layer["name"]
                items = layer["items"]
                layer_points = {}
                item_points = []
                item_bounds = []
                if len(items) > 0:
                    for item in items:
                        bounds = item["bounds"]
                        x, y = bounds["x"], bounds["y"]
                        w, h = bounds["width"], bounds["height"]
                        item_bounds.append((x, y, x + w, y + h))
                        segment_points = []
                        segments = item["segments"]
                        for segment in segments:
                            x = segment["point"]["x"]
                            y = segment["point"]["y"]
                            segment_points.append([x, y])
                        item_points.append(segment_points)
                num_points += len(item_points)
                layer_points.update({"bounds": item_bounds})
                layer_points.update({"points": item_points})
                dict_points.update({name: layer_points}) 
            if num_points > 0:
                with save_json_path.open("w") as handle:
                    json.dump(dict_points, handle)   
    return


def main():
    csv_dir = "/well/rittscher/shared/datasets/KiBla/data"
    keyword = "Bladder"
    save_json_dir = f"a_06semantic_segmentation/wsi_{keyword.lower()}_annotations"
    csv_paths = pathlib.Path(csv_dir).glob("*.csv")
    for csv_path in csv_paths:
        csv_name = csv_path.stem
        if keyword in csv_name and "B1" not in csv_name:
            csv_to_json(csv_path, save_json_dir)
    return

if __name__ == "__main__":
    main()