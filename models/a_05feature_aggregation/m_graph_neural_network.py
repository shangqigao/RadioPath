import json
import joblib
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm1d, Linear, ReLU
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn import EdgeConv, GINConv, GCNConv, GATConv


class SlideGraphDataset(Dataset):
    """loading graph data from disk
    """
    def __init__(self, info_list, mode="train", preproc=None):
        super().__init__()
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
    
    def get(self, idx):
        info = self.info_list[idx]
        if any(v in self.mode for v in ["train", "valid"]):
            wsi_path, label_path = info
            label = torch.tensor(np.load(f"{label_path}"))
        else:
            wsi_path = info
        
        with pathlib.Path(wsi_path).open() as fptr:
            graph_dict = json.load(fptr)
        graph_dict = {k: np.array(v) for k, v in graph_dict.items() if k != "cluster_points"}

        if self.preproc is not None:
            graph_dict["x"] = self.preproc(graph_dict["x"])

        graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
        if any(v in self.mode for v in ["train", "valid"]):
            graph_dict.update({"y": label})
        graph = Data(**graph_dict)
        return graph
    
    def len(self):
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
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.conv_name = conv

        conv_dict = {
            "MLP": [Linear, 1],
            "GCNConv": [GCNConv, 1],
            "GATConv": [GATConv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                BatchNorm1d(out_dims),
                ReLU(),
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        self.tail = {
            "MLP": Linear(self.embedding_dims[-1], dim_target),
            "GCNConv": GCNConv(self.embedding_dims[-1], dim_target),
            "GATConv": GATConv(self.embedding_dims[-1], dim_target),
            "GINConv": Linear(self.embedding_dims[-1], dim_target),
            "EdgeConv": Linear(self.embedding_dims[-1], dim_target)
        }[self.conv_name]

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embedding_dims[1:]:
            conv_class, alpha = conv_dict[self.conv_name]
            if self.conv_name in ["GINConv", "EdgeConv"]:
                block = create_block(alpha * input_emb_dim, out_emb_dim)
                subnet = conv_class(block, **kwargs)
                self.convs.append(subnet)
                self.linears.append(Linear(out_emb_dim, out_emb_dim))
            elif self.conv_name in ["GCNConv", "GATConv"]:
                subnet = conv_class(alpha * input_emb_dim, out_emb_dim)
                self.convs.append(subnet)
                self.linears.append(create_block(out_emb_dim, out_emb_dim))
            else:
                subnet = create_block(alpha * input_emb_dim, out_emb_dim)
                self.convs.append(subnet)
                self.linears.append(nn.Sequential())
                
            input_emb_dim = out_emb_dim

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

        feature = self.head(feature)
        for layer in range(1, self.num_layers):
            feature = F.dropout(feature, p=self.dropout, training=self.training)
            if self.conv_name in ["MLP"]:
                feature = self.convs[layer - 1](feature)
            else:
                feature = self.convs[layer - 1](feature, edge_index)
            feature = self.linears[layer - 1](feature)
        if self.conv_name in ["MLP", "GINConv", "EdgeConv"]:
            feature = self.tail(feature)
        else:
            feature = self.tail(feature, edge_index)
        return feature
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer, probs=None, tau=1):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.train()
        optimizer.zero_grad()
        wsi_outputs = model(wsi_graphs)
        ## using logit ajusted cross-entropy
        if probs is not None:
            shift = tau * torch.log(torch.tensor(probs) + 1e-12)
            wsi_outputs = wsi_outputs + shift.to(device)
        wsi_outputs = wsi_outputs.squeeze()
        wsi_labels = wsi_graphs.y.squeeze()
        loss = loss(wsi_outputs, wsi_labels)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = wsi_labels.cpu().numpy()
        return [loss, wsi_outputs, wsi_labels]
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.eval()
        with torch.inference_mode():
            wsi_outputs = model(wsi_graphs)
        wsi_outputs = wsi_outputs.squeeze().cpu().numpy()
        if wsi_graphs.y is not None:
            wsi_labels = wsi_graphs.y.squeeze().cpu().numpy()
            postive = wsi_labels > 0
            if wsi_outputs.ndim == 1:
                wsi_outputs = wsi_outputs
                wsi_labels = np.array(postive, dtype=np.int32)
            else:
                wsi_outputs = wsi_outputs[postive, :]
                wsi_labels = wsi_labels[postive] - 1
            return [wsi_outputs, wsi_labels]
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

class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, mode="nnPU"):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.mode = mode
        self.unlabeled = 0
        self.min_count = torch.tensor(1.)
    
    def forward(self, input, target):   
        assert input.shape == target.shape, f"input shape is {input.shape}, but target shape is {target.shape}"  
        positive, unlabeled = target > self.unlabeled, target == self.unlabeled
        
        y_positive = self.loss_func(input)
        y_unlabeled = self.loss_func(-input)

        positive_risk = self.prior * torch.mean(y_positive[positive])
        negative_risk = torch.mean(y_unlabeled[unlabeled]) - self.prior * torch.mean(y_unlabeled[positive])

        if self.mode == "nnPU":
            if negative_risk < -self.beta:
                return -self.gamma * negative_risk
            else:
                return positive_risk + negative_risk
        elif self.mode == "uPU":
            return positive_risk + negative_risk
        elif self.mode == "PN":
            negative_risk = (1 - self.prior) * torch.mean(y_unlabeled[unlabeled])
            return positive_risk + negative_risk
        else:
            raise ValueError(f"{self.mode} is not supported!")
