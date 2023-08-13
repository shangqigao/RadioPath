import csv
import json
import os
import math
import matplotlib.pyplot as plt

csv_dir = "/well/rittscher/shared/datasets/KiBla/data"
csv_path = os.path.join(csv_dir, "KIBLA_Bladder_B4_Clare_Annotations_2023_07_31_16_04.csv")
# csv_path = os.path.join(csv_dir, "Bladder/Annotations/Exported_Annotations_KIBLA_Bladder_B1.csv")
json_path = "output.json"
tile_size = 256

def count_annotated_tiles():
    data = {}
    with open(csv_path) as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            dict_ann = json.loads(row["Annotation"])
            layers = dict_ann["annotation"]["layers"]
            for layer in layers:
                name = layer["name"]
                items = layer["items"]
                num_tiles = 0.
                if len(items) > 0:
                    for item in items:
                        bbox_height = item["bounds"]["height"]
                        bbox_width = item["bounds"]["width"]
                        num_tiles += bbox_height * bbox_width / tile_size ** 2
                if i == 0:
                    data[name] = [num_tiles]
                else:
                    data[name].append(num_tiles)
        
    return data

def visualize_class(data):
    keys, values = [], []
    for k, v in data.items():
        keys.append(k)
        values.append(int(sum(v)))
    plt.figure()
    plt.subplot(211)
    plt.bar(keys, values)
    plt.ylabel('Number of tiles')
    plt.title('Kidney bar')
    plt.xticks()
    plt.subplot(212)
    plt.pie(values, labels=keys, autopct='%.2f%%')
    plt.title('Kidney pie')
    plt.savefig("annotation_analysis.jpg")
    return

def main():
    data = count_annotated_tiles()
    visualize_class(data)
    return

if __name__ == "__main__":
    main()