import json
import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm1d, Linear, ReLU
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn import EdgeConv, GINConv, global_add_pool, global_max_pool, global_mean_pool

class SlideGraphDataset(Dataset):
    """loading graph data from disk
    """
    def __init__(self, info_list, mode="train", preproc=None):
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc

    def __getitem__(self, idx):
        info = self.info_list[idx]
        if any(v in self.mode for v in ["train", "valid"]):
            wsi_path, label_path = info
            label = torch.tensor(np.load(f"{label_path}"))
        else:
            wsi_path = info
        
        with wsi_path.open() as fptr:
            graph_dict = json.load(fptr)
        graph_dict = {k: np.array(v) for k, v in graph_dict.items()}

        if self.preproc is not None:
            graph_dict["x"] = self.preproc(graph_dict["x"])

        graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
        graph = Data(**graph_dict)

        if any(v in self.mode for v in ["train", "valid"]):
            return {"graph": graph, "label": label}
        return {"graph": graph}
    
    def __len__(self):
        return len(self.info_list)
    
class SlideGraphArch(nn.Module):
    """define SlideGraph architecture
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers=None,
            dropout=0.0,
            conv="GINConv",
            **kwargs,
    ):
        super().__init__()
        if layers is None:
            layers = [6, 6]
        self.dropout = dropout
        self.embedding_dims = layers
        self.num_layers = len(self.embedding_dims)
        self.nns = []
        self.convs = []
        self.linears = []

        conv_dict = {"GINConv": [GINConv, 1], "EdgeConv": [EdgeConv, 2]}
        if conv not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_linear(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                BatchNorm1d(out_dims),
                ReLU(),
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.first_h = create_linear(input_emb_dim, out_emb_dim)
        self.linears.append(Linear(out_emb_dim, dim_target))

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embedding_dims[1:]:
            conv_class, alpha = conv_dict[conv]
            subnet = create_linear(alpha * input_emb_dim, out_emb_dim)
            self.nns.append(subnet)
            self.convs.append(conv_class(self.nns[-1], **kwargs))
            self.linears.append(Linear(out_emb_dim, dim_target))
            input_emb_dim = out_emb_dim

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)
        self.aux_model = {}

    def save(self, path, aux_path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        joblib.dump(self.aux_model, aux_path)

    def load(self, path, aux_path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.aux_model = joblib.load(aux_path)

    def forward(self, data):
        feature, edge_index, batch = data.x, data.edge_index, data.batch

        node_prediction = 0

        feature = self.first_h(feature)
        for layer in range(self.num_layers):
            if layer == 0:
                node_prediction_sub = self.linears[layer](feature)
                node_prediction += node_prediction_sub
                node_prediction = F.dropout(
                    node_prediction,
                    p=self.dropout,
                    training=self.training,
                )
            else:
                feature = self.convs[layer - 1](feature, edge_index)
                node_prediction_sub = self.linears[layer](feature)
                node_prediction += node_prediction_sub
                node_prediction = F.dropout(
                    node_prediction,
                    p=self.dropout,
                    training=self.training,
                )
        return F.log_softmax(node_prediction, dim=1)
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, optimizer):
        wsi_graphs = batch_data["graph"].to("cuda")
        wsi_labels = batch_data["label"].to("cuda")

        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.train()
        optimizer.zero_grad()
        wsi_outputs = model(wsi_graphs)
        loss = F.nll_loss(wsi_outputs, wsi_labels)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = wsi_labels.cpu().numpy()
        return [loss, wsi_outputs, wsi_labels]
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        wsi_graphs = batch_data["graph"].to("cuda")
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.eval()
        with torch.inference_mode():
            wsi_outputs = model(wsi_graphs)
        
        wsi_outputs = wsi_outputs.cpu().numpy()
        wsi_outputs = np.transpose(wsi_outputs, (0, 2, 1))
        wsi_outputs = np.reshape(wsi_outputs, (-1, wsi_outputs.shape[2]))
        if "label" in batch_data:
            wsi_labels = batch_data["label"]
            wsi_labels = wsi_labels.cpu().numpy()
            wsi_labels = np.reshape(wsi_labels, (-1, 1))
            return wsi_outputs, wsi_labels
        return [wsi_outputs]
    

class ScalarMovingAverage:
    """Class to calculate running average."""

    def __init__(self, alpha=0.95):
        """Initialize ScalarMovingAverage."""
        super().__init__()
        self.alpha = alpha
        self.tracking_dict = {}

    def __call__(self, step_output):
        """ScalarMovingAverage instances behave and can be called like a function."""
        for key, current_value in step_output.items():
            if key in self.tracking_dict:
                old_ema_value = self.tracking_dict[key]
                # Calculate the exponential moving average
                new_ema_value = (
                    old_ema_value * self.alpha + (1.0 - self.alpha) * current_value
                )
                self.tracking_dict[key] = new_ema_value
            else:  # Init for variable which appear for the first time
                new_ema_value = current_value
                self.tracking_dict[key] = new_ema_value
    
