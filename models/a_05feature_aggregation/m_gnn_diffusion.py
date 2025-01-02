import json
import joblib
import pathlib
import numpy as np

import torch
import torch.nn as nn


import torch.utils
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_dense_adj

class SlideGraphSpectrumDataset(Dataset):
    """loading graph data from disk
    """
    def __init__(self, info_list, mode="train", preproc=None, max_num_nodes=2048):
        super().__init__()
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
    
    def get(self, idx):
        info = self.info_list[idx]
        graph_path = info
        
        with pathlib.Path(graph_path).open() as fptr:
            graph_dict = json.load(fptr)
        graph_dict = {k: np.array(v) for k, v in graph_dict.items() if k != "cluster_points"}

        if self.preproc is not None:
            graph_dict["x"] = self.preproc(graph_dict["x"])

        graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
        graph = Data(**graph_dict)
        return graph
    
    def len(self):
        return len(self.info_list)
    
    
class SlideGraphSpectrumDiffusionArch(nn.Module):
    """define SlideGraph architecture
    """
    def __init__(
            self, 
            config,
            params_x,
            params_adj,
            model_x,
            model_adj
    ):
        super().__init__()
        self.config = config
        self.params_x = params_x
        self.params_adj = params_adj
        self.model_x = model_x
        self.model_adj = model_adj
        
    def save(self, x_path, adj_path):
        x_state_dict = self.model_x.state_dict()
        torch.save(x_state_dict, x_path)
        adj_state_dict = self.model_adj.state_dict()
        torch.save(adj_state_dict, adj_path)

    def load(self, x_path, adj_path):
        x_state_dict = torch.load(x_path)
        self.model_x.load_state_dict(x_state_dict)
        adj_state_dict = torch.load(adj_path)
        self.model_adj.load_state_dict(adj_state_dict)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        adj = to_dense_adj(edge_index, max_num_nodes=x.size(0))
        la, u = torch.linalg.eigh(adj)
        subjects = (x, adj, u, la)
        return subjects
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizers):
        device = "cuda" if on_gpu else "cpu"
        train_graphs = batch_data.to(device)

        train_graphs.x = train_graphs.x.type(torch.float32)

        model.model_x.train()
        model.model_adj.train()
        optimizer_x, optimizer_adj = optimizers
        optimizer_x.zero_grad()
        optimizer_adj.zero_grad()

        subjects = model(train_graphs)
        loss_x, loss_adj = loss(model.model_x, model.model_adj, *subjects) 
        loss_x.backward()
        loss_adj.backward()

        nn.utils.clip_grad_norm_(model.model_x.parameters(), model.config.train.grad_norm)
        nn.utils.clip_grad_norm_(model.model_adj.parameters(), model.config.train.grad_norm)
        optimizer_x.step()
        optimizer_adj.step()

        loss_x = loss_x.detach().cpu().numpy()
        assert not np.isnan(loss_x)
        loss_adj = loss_adj.detach().cpu().numpy()
        assert not np.isnan(loss_adj)

        return [loss_x, loss_adj]
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu, loss):
        device = "cuda" if on_gpu else "cpu"
        infer_graphs = batch_data.to(device)
        infer_graphs.x = infer_graphs.x.type(torch.float32)

        model.model_x.eval()
        model.model_adj.eval()
        with torch.inference_mode():
            subjects = model(infer_graphs)
            loss_x, loss_adj = loss(model.model_x, model.model_adj, *subjects) 

        return [loss_x.item(), loss_adj.item()]

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
