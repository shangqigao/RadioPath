import pathlib
import json
import numpy as np

root_dir = '/home/sg2162/rds/hpc-work/Experiments/pathomics/TCGA-RCC_wsi_pathomic_features/conch'

feature_paths = pathlib.Path(root_dir).glob('*.features.npy')
feature_paths = [f"{p}" for p in feature_paths]
num_features = []
for i, p in enumerate(feature_paths):
    print(f"Loading [{i+1}/{len(feature_paths)}]")
    # with open(p, 'r') as file:
    #     data = json.load(file)
    #     num_features.append(len(data['x']))
    data = np.load(p)
    num_features.append(len(data))
print("Average per sample:", sum(num_features) / len(num_features))