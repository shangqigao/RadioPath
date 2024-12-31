import json
import joblib
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchbnn as bnn

from torch.nn import BatchNorm1d, Linear, ReLU, Dropout
from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph, softmax
from torch_geometric.nn import EdgeConv, GINConv, GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, AttentionalAggregation



class SlideGraphDataset(Dataset):
    """loading graph data from disk
    """
    def __init__(self, info_list, mode="train", preproc=None, subgraph=False, PN=None):
        super().__init__()
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
        self.subgraph = subgraph
        self.PN = PN
    
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
            if self.subgraph:
                positive = label.squeeze() > 0  # 0 is unlabeled
                # graph_dict["x"] = graph_dict["x"][positive]
                edge_index, _ = subgraph(positive, graph_dict["edge_index"], relabel_nodes=False)
                graph_dict["edge_index"] = edge_index
                # label = label[positive]
            graph_dict.update({"y": label})
        else:
            if self.subgraph:
                assert len(self.PN[idx]) == len(graph_dict["x"]), "Must be one-to-one labels"
                positive = torch.tensor(self.PN[idx]).squeeze() > 0
                # graph_dict["x"] = graph_dict["x"][positive]
                edge_index, _ = subgraph(positive, graph_dict["edge_index"], relabel_nodes=False)
                graph_dict["edge_index"] = edge_index

        graph = Data(**graph_dict)
        return graph
    
    def len(self):
        return len(self.info_list)
    
class SurvivalGraphDataset(Dataset):
    """loading graph data for survival analysis
    """
    def __init__(self, info_list, mode="train", preproc=None, 
                 data_types=["radiomics", "pathomics"],
                 sampling_rate=1.0):
        super().__init__()
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
        self.data_types = data_types
        self.sampling_rate = sampling_rate
    
    def get(self, idx):
        info = self.info_list[idx]
        if any(v in self.mode for v in ["train", "valid"]):
            graph_path, label = info
            label = torch.tensor(label).unsqueeze(0)
        else:
            graph_path = info
        
        if len(self.data_types) == 1:
            key = self.data_types[0]
            with pathlib.Path(graph_path[key]).open() as fptr:
                graph_dict = json.load(fptr)
            graph_dict = {k: np.array(v) for k, v in graph_dict.items() if k != "cluster_points"}
            if key == "radiomics": 
                graph_dict["edge_index"] = graph_dict["edge_index"].T

            if self.preproc is not None:
                graph_dict["x"] = self.preproc[key](graph_dict["x"])

            graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
            graph_dict["edge_index"] = graph_dict["edge_index"].type(torch.int64)
            if any(v in self.mode for v in ["train", "valid"]):
                graph_dict.update({"y": label})

            graph = Data(**graph_dict)
            if self.sampling_rate < 1 and key == "radiomics":
                num_nodes = len(graph_dict["x"])
                num_sampled = int(num_nodes*self.sampling_rate)
                subset = np.random.choice(num_nodes, num_sampled, replace=False)
                loader = NeighborLoader(
                    data=graph,
                    num_neighbors=[3, 2],
                    input_nodes=subset,
                    batch_size=1
                )
                graph = next(iter(loader))
        else:
            graph_dict = {}
            for key in self.data_types:
                with pathlib.Path(graph_path[key]).open() as fptr:
                    subgraph_dict = json.load(fptr)
                subgraph_dict = {k: np.array(v) for k, v in subgraph_dict.items() if k != "cluster_points"}
                if key == "radiomics": subgraph_dict["edge_index"] = subgraph_dict["edge_index"].T

                if self.preproc is not None:
                    subgraph_dict["x"] = self.preproc[key](subgraph_dict["x"])

                subgraph_dict = {k: torch.tensor(v) for k, v in subgraph_dict.items()}
                if any(v in self.mode for v in ["train", "valid"]):
                    subgraph_dict.update({"y": label})

                edge_dict = {"edge_index": subgraph_dict["edge_index"].type(torch.int64)}
                subgraph_dict = {k: v.type(torch.float32) for k, v in subgraph_dict.items() if k != "edge_index"}
                graph_dict.update({key: subgraph_dict, (key, "to", key): edge_dict})
            
            graph = HeteroData(graph_dict)
            if self.sampling_rate < 1 and "radiomics" in self.data_types:
                num_nodes = len(graph_dict["radiomics"]["x"])
                num_sampled = int(num_nodes*self.sampling_rate)
                subset_tuple = ("radiomics", np.random.choice(num_nodes, num_sampled, replace=False))
                loader = NeighborLoader(
                    data=graph,
                    num_neighbors={"radiomics": [3, 2]},
                    input_nodes=subset_tuple,
                    batch_size=1
                )  
                graph = next(iter(loader))
        return graph
    
    def len(self):
        return len(self.info_list)
    
class ConceptGraphDataset(Dataset):
    """loading graph data for concept learning
    """
    def __init__(self, info_list, mode="train", preproc=None, data_types=["pathomics"]):
        super().__init__()
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
        self.data_types = data_types
    
    def get(self, idx):
        info = self.info_list[idx]
        if any(v in self.mode for v in ["train", "valid"]):
            graph_path, y = info
            concept = torch.tensor(y["concept"]).unsqueeze(0)
            label = torch.tensor(y["label"]).unsqueeze(0)
        else:
            graph_path = info
        
        if len(self.data_types) == 1:
            key = self.data_types[0]
            with pathlib.Path(graph_path[key]).open() as fptr:
                graph_dict = json.load(fptr)
            graph_dict = {k: np.array(v) for k, v in graph_dict.items() if k != "cluster_points"}
            if key == "radiomics": graph_dict["edge_index"] = graph_dict["edge_index"].T

            if self.preproc is not None:
                graph_dict["x"] = self.preproc[key](graph_dict["x"])

            graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
            if any(v in self.mode for v in ["train", "valid"]):
                graph_dict.update({"concept": concept, "y": label})
            
            graph_dict = {k: v.type(torch.float32) for k, v in graph_dict.items()}
            graph_dict.update({"edge_index": graph_dict["edge_index"].type(torch.int64)})
            graph = Data(**graph_dict)
        else:
            graph_dict = {}
            for key in self.data_types:
                with pathlib.Path(graph_path[key]).open() as fptr:
                    subgraph_dict = json.load(fptr)
                subgraph_dict = {k: np.array(v) for k, v in subgraph_dict.items() if k != "cluster_points"}
                if key == "radiomics": subgraph_dict["edge_index"] = subgraph_dict["edge_index"].T
                if self.preproc is not None:
                    subgraph_dict["x"] = self.preproc[key](subgraph_dict["x"])

                subgraph_dict = {k: torch.tensor(v) for k, v in subgraph_dict.items()}
                if any(v in self.mode for v in ["train", "valid"]):
                    subgraph_dict.update({"concept": concept, "y": label})

                edge_dict = {"edge_index": subgraph_dict["edge_index"].type(torch.int64)}
                subgraph_dict = {k: v.type(torch.float32) for k, v in subgraph_dict.items() if k != "edge_index"}
                graph_dict.update({key: subgraph_dict, (key, "to", key): edge_dict})
            
            graph = HeteroData(graph_dict)
        return graph
    
    def len(self):
        return len(self.info_list)

class SubtypingGraphDataset(Dataset):
    """loading graph data for cancer subtyping
    """
    def __init__(self, info_list, mode="train", preproc=None):
        super().__init__()
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
    
    def get(self, idx):
        info = self.info_list[idx]
        if any(v in self.mode for v in ["train", "valid"]):
            wsi_path, label = info
            label = torch.tensor(label)
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
            "GATConv": GATv2Conv(self.embedding_dims[-1], dim_target),
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
            output = self.tail(feature)
        else:
            output = self.tail(feature, edge_index)
        return output, feature
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, kl, optimizer, probs=None, tau=1, centroid=None):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.train()
        optimizer.zero_grad()
        wsi_outputs, feature = model(wsi_graphs)
        ## using logit ajusted cross-entropy
        if probs is not None:
            shift = tau * torch.log(torch.tensor(probs) + 1e-12)
            wsi_outputs = wsi_outputs + shift.to(device)
        wsi_outputs = wsi_outputs.squeeze()
        wsi_labels = wsi_graphs.y.squeeze()
        loss = loss(wsi_outputs, wsi_labels)
        ## using contrastive loss for positive samples
        if centroid is not None:
            centroid = torch.from_numpy(centroid).to(device)
            k = centroid.shape[2]
            feature = feature[wsi_labels > 0]
            expand_feat = feature.unsqueeze(2).expand(-1, -1, k)
            fro_norm = torch.norm(expand_feat - centroid, dim=1)
            clusters = torch.argmin(fro_norm, dim=1)
            feat_norm = torch.norm(feature, dim=1, keepdim=True)
            feat_prod = torch.matmul(feature, feature.transpose(0, 1))
            cos_sim = feat_prod / torch.matmul(feat_norm, feat_norm.transpose(0, 1))
            intra_sim = []
            c = feature.shape[1]
            intra_feat_sum = torch.zeros(size=(1, c, k)).to(device)
            intra_mem_num = torch.zeros(size=(1, 1, k)).to(device)
            for i in range(k):
                indecies = clusters == i
                if len(indecies) > 0:
                    intra_feature = feature[indecies, :]
                    intra_feat_sum[..., i] = intra_feature.sum(axis=0, keepdim=True)
                    intra_mem_num[0, 0, i] = len(indecies)
                    intra_sim_i = cos_sim[indecies, indecies]
                    intra_sim.append(torch.exp(intra_sim_i).sum())
            contrastive_loss = - sum(intra_sim) / torch.exp(cos_sim).sum()
            loss = loss + contrastive_loss
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = wsi_labels.cpu().numpy()
        if centroid is not None:
            intra_feat_sum = intra_feat_sum.detach().cpu().numpy()
            intra_mem_num = intra_mem_num.detach().cpu().numpy()
            return [loss, intra_feat_sum, intra_mem_num]
        else:
            return [loss, wsi_outputs, wsi_labels]
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu, mode):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.eval()
        with torch.inference_mode():
            wsi_outputs, feature = model(wsi_graphs)
        wsi_outputs = wsi_outputs.squeeze().cpu().numpy()
        feature = feature.squeeze().cpu().numpy()
        if wsi_graphs.y is not None:
            wsi_labels = wsi_graphs.y.squeeze().cpu().numpy()

            ## convert outputs and labels based given mode
            if mode in ["PN", "uPU", "nnPU"]:
                wsi_outputs = wsi_outputs
                wsi_labels = np.array(wsi_labels > 0, dtype=np.int32)
            elif mode == "PCE":
                postive = wsi_labels > 0
                wsi_outputs = wsi_outputs[postive]
                wsi_labels = wsi_labels[postive] - 1
            elif mode == "LHCE":
                low = wsi_labels == 1
                high = wsi_labels == 3
                low_outputs, low_labels = wsi_outputs[low], wsi_labels[low] - 1
                high_outputs, high_labels = wsi_outputs[high], wsi_labels[high] - 2
                wsi_outputs = np.concatenate([low_outputs, high_outputs], axis=0)
                wsi_labels = np.concatenate([low_labels, high_labels], axis=0)
            return [wsi_outputs, wsi_labels]
        return [wsi_outputs, feature]

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
    
class Bayes_Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1, Bayes_std=0.1):
        super(Bayes_Attn_Net_Gated, self).__init__()
        self.attention_a = [
            # bnn.BayesLinear(0, Bayes_std, L, D),
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [
            # bnn.BayesLinear(0, Bayes_std, L, D),
            nn.Linear(L, D),
            nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = bnn.BayesLinear(0, Bayes_std, D, n_classes)
        # self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
    
class Score_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Score_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D)]

        self.attention_b = [nn.Linear(L, D)]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
    
class ImportanceScoreArch(nn.Module):
    """define importance score architecture
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers=None,
            dropout=0.0,
            conv="GINConv",
            mode="encoder",
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
            "GATConv": [GATv2Conv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                # BatchNorm1d(out_dims),
                ReLU(),
                Dropout(self.dropout)
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        if mode == "encoder":
            self.tail = nn.Sequential(
                Linear(self.embedding_dims[-1], dim_target),
                BatchNorm1d(dim_target),
            )
        elif mode == "decoder":
            self.tail = Linear(self.embedding_dims[-1], dim_target)  

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

    def forward(self, feature, edge_index):
        feature = self.head(feature)
        for layer in range(1, self.num_layers):
            if self.conv_name in ["MLP"]:
                feature = self.convs[layer - 1](feature)
            else:
                feature = self.convs[layer - 1](feature, edge_index)
            feature = self.linears[layer - 1](feature)
        output = self.tail(feature)
        return output

class SurvivalGraphArch(nn.Module):
    """define Graph architecture for survival analysis
    Args:
        aggregation: attention-based multiple instance learning (ABMIL) 
        or stochastic importance score informed regression (SISIR)
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers=None,
            dropout=0.0,
            conv="GINConv",
            keys=["radiomics", "pathomics"],
            aggregation="SISIR",
            **kwargs,
    ):
        super().__init__()
        if layers is None:
            layers = [6, 6]
        self.dropout = dropout
        self.embedding_dims = layers
        self.keys = keys
        self.aggregation = aggregation
        self.num_layers = len(self.embedding_dims)
        self.enc_convs = nn.ModuleList()
        self.enc_linears = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.dec_linears = nn.ModuleList()
        self.conv_name = conv
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                BatchNorm1d(out_dims),
                ReLU(),
                Dropout(self.dropout)
            )
        
        if len(keys) > 1:
            out_emb_dim = self.embedding_dims[0]
            self.enc_branches = nn.ModuleDict({k: create_block(dim_features[k], out_emb_dim) for k in keys})
            self.dec_branches = nn.ModuleDict({k: create_block(out_emb_dim, dim_features[k]) for k in keys})
            input_emb_dim = out_emb_dim
            out_emb_dim = self.embedding_dims[0]
        elif len(keys) == 1:
            input_emb_dim = dim_features[keys[0]]
            out_emb_dim = self.embedding_dims[0]
        else:
            raise NotImplementedError

        self.head = create_block(input_emb_dim, out_emb_dim)
        input_emb_dim = out_emb_dim

        if aggregation == "ABMIL":
            self.gate_nn = Attn_Net_Gated(
                L=input_emb_dim,
                D=256,
                dropout=0.25,
                n_classes=1
            )
            self.Aggregation = SumAggregation()
        elif aggregation == "SISIR":
            # out_emb_dim = self.embedding_dims[-1]
            self.score_nn = ImportanceScoreArch(
                dim_features=input_emb_dim,
                dim_target=2,
                layers=self.embedding_dims[1:],
                dropout=self.dropout,
                conv=self.conv_name,
                mode="decoder"
            )
            # self.gate_nn = Attn_Net_Gated(
            #     L=out_emb_dim,
            #     D=64,
            #     dropout=0.0,
            #     n_classes=2
            # )
            self.inverse_score_nn = ImportanceScoreArch(
                dim_features=2,
                dim_target=input_emb_dim,
                layers=self.embedding_dims[::-1],
                dropout=self.dropout,
                conv=self.conv_name,
                mode="decoder"
            )
            self.Aggregation = MeanAggregation()
            # input_emb_dim = out_emb_dim
        else:
            raise NotImplementedError
        self.classifier = Linear(input_emb_dim, dim_target)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def sampling(self, data):
        assert data.size(-1) == 2
        loc, logvar = data[..., 0:1], data[..., 1:2]
        logvar = torch.clamp(logvar, min=-20, max=20)
        gauss = torch.distributions.Normal(loc, torch.exp(0.5*logvar))
        return gauss.sample(), [loc, logvar]

    def forward(self, data):
        if len(self.keys) > 1:
            x_dict = data.x_dict
            edge_index_dict = data.edge_index_dict
            batch_dict = data.batch_dict
            enc_list = [x_dict[k] for k in self.keys]
            feature_list = [self.enc_branches[k](enc) for k, enc in zip(self.keys, enc_list)]
            feature = torch.concat(feature_list, dim=0)
            edge_index_list = [edge_index_dict[k, "to", k] for k in self.keys]
            # check = [[len(f), e.max().item()] for f, e in zip(feature_list, edge_index_list)]
            # print("Number of nodes and edge index: ", check)
            index_shift = 0
            for i in range(1, len(edge_index_list)): 
                index_shift += len(feature_list[i-1])
                edge_index_list[i] += index_shift
            edge_index = torch.concat(edge_index_list, dim=1)
            batch_list = [batch_dict[k] for k in self.keys]
            batch = torch.concat(batch_list, dim=0)
        else:
            enc_list = [data.x]
            feature, edge_index, batch = data.x, data.edge_index, data.batch

        feature = self.head(feature)
        if self.aggregation == "ABMIL":
            gate = self.gate_nn(feature)
            gate = softmax(gate, index=batch, dim=-2)
            VIparas = None
        elif self.aggregation == "SISIR":
            # encoder
            encode = self.score_nn(feature, edge_index)
            # encode = self.gate_nn(feature)
            # reparameterization
            gate, VIparas = self.sampling(encode)
            # decoder
            decode = self.inverse_score_nn(encode, edge_index)

            if len(self.keys) > 1:
                dec_list, index_s, index_e = [], 0, 0
                for i, k in enumerate(self.keys):
                    index_e += len(enc_list[i])
                    dec_list.append(self.dec_branches[k](decode[index_s:index_e, ...]))
                    index_s = index_e
            else:
                dec_list = [decode]
            VIparas.append(enc_list)
            VIparas.append(dec_list)
        feature = self.Aggregation(feature * gate, index=batch)
        output = self.classifier(feature)
        return output, VIparas
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer, kl=None):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        model.train()
        optimizer.zero_grad()
        wsi_outputs, VIparas = model(wsi_graphs)
        wsi_outputs = wsi_outputs.squeeze()
        if hasattr(wsi_graphs, "y_dict"):
            wsi_labels = wsi_graphs.y_dict[model.keys[0]].squeeze()
        elif hasattr(wsi_graphs, "y"):
            wsi_labels = wsi_graphs.y.squeeze()
        else:
            raise AttributeError
        loss = loss(wsi_outputs, wsi_labels[:, 0], wsi_labels[:, 1], VIparas)
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

        model.eval()
        with torch.inference_mode():
            wsi_outputs, _ = model(wsi_graphs)
        wsi_outputs = wsi_outputs.cpu().numpy()
        wsi_labels = None
        if hasattr(wsi_graphs, "y_dict"):
            wsi_labels = wsi_graphs.y_dict[model.keys[0]].cpu().numpy()
        elif hasattr(wsi_graphs, "y"):
            wsi_labels = wsi_graphs.y.cpu().numpy()
        return [wsi_outputs, wsi_labels]

class SurvivalBayesGraphArch(nn.Module):
    """define Graph architecture for survival analysis
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers=None,
            dropout=0.0,
            conv="GINConv",
            Bayes_std=0.1,
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
            "MLP": [None, 1],
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
                ReLU(),
                Dropout(self.dropout)
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        input_emb_dim = out_emb_dim
        out_emb_dim = self.embedding_dims[-1]

        # input_emb_dim = out_emb_dim
        # for out_emb_dim in self.embedding_dims[1:]:
        #     conv_class, alpha = conv_dict[self.conv_name]
        #     if self.conv_name in ["GINConv", "EdgeConv"]:
        #         block = create_block(alpha * input_emb_dim, out_emb_dim)
        #         subnet = conv_class(block, **kwargs)
        #         self.convs.append(subnet)
        #         self.linears.append(bnn.BayesLinear(0, Bayes_std, input_emb_dim, out_emb_dim))
        #     elif self.conv_name in ["GCNConv", "GATConv"]:
        #         subnet = conv_class(alpha * input_emb_dim, out_emb_dim)
        #         self.convs.append(subnet)
        #         self.linears.append(create_block(out_emb_dim, out_emb_dim))
        #     else:
        #         subnet = create_block(alpha * input_emb_dim, out_emb_dim)
        #         self.convs.append(subnet)
        #         self.linears.append(nn.Sequential())
                
        #     input_emb_dim = out_emb_dim
            
        self.gate_nn = Bayes_Attn_Net_Gated(
            L=input_emb_dim,
            D=out_emb_dim,
            dropout=0.25,
            n_classes=1,
            Bayes_std=Bayes_std
        )
        self.global_attention = AttentionalAggregation(
            gate_nn=self.gate_nn,
            nn=None
        )
        self.classifier = Linear(input_emb_dim, dim_target)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def forward(self, data):
        feature, edge_index, batch = data.x, data.edge_index, data.batch

        feature = self.head(feature)
        # for layer in range(1, self.num_layers):
        #     feature = F.dropout(feature, p=self.dropout, training=self.training)
        #     if self.conv_name in ["MLP"]:
        #         feature = self.convs[layer - 1](feature)
        #     else:
        #         feature = self.convs[layer - 1](feature, edge_index)
        #     feature = self.linears[layer - 1](feature)

        feature = self.global_attention(feature, index=batch)
        output = self.classifier(feature)
        return output
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer, kl):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.train()
        optimizer.zero_grad()
        wsi_outputs = model(wsi_graphs)
        wsi_outputs = wsi_outputs.squeeze()
        wsi_labels = wsi_graphs.y.squeeze()
        loss = loss(wsi_outputs, wsi_labels[:, 0], wsi_labels[:, 1])
        kl_loss, kl_weight = kl["loss"](model)[0], kl["weight"]
        loss = loss + kl_weight * kl_loss
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
        wsi_outputs = wsi_outputs.cpu().numpy()
        if wsi_graphs.y is not None:
            wsi_labels = wsi_graphs.y.cpu().numpy()

            return [wsi_outputs, wsi_labels]
        return [wsi_outputs]

class ConceptScoreArch(nn.Module):
    """define concept score architecture
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
            "GATConv": [GATv2Conv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                ReLU(),
                Dropout(self.dropout)
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        self.tail = {
            "MLP": Linear(self.embedding_dims[-1], dim_target),
            "GCNConv": GCNConv(self.embedding_dims[-1], dim_target),
            "GATConv": GATv2Conv(self.embedding_dims[-1], dim_target),
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

    def forward(self, feature, edge_index):
        feature = self.head(feature)
        for layer in range(1, self.num_layers):
            if self.conv_name in ["MLP"]:
                feature = self.convs[layer - 1](feature)
            else:
                feature = self.convs[layer - 1](feature, edge_index)
            feature = self.linears[layer - 1](feature)
        if self.conv_name in ["MLP", "GINConv", "EdgeConv"]:
            output = self.tail(feature)
        else:
            output = self.tail(feature, edge_index)
        return output

class ConceptGraphArch(nn.Module):
    """define Graph architecture for concept learning
    Args:
        aggregation: attention-based multiple instance learning (ABMIL) 
        or concept bottleneck model (CBM)
    """
    def __init__(
            self, 
            dim_features, 
            dim_concept,
            dim_target,
            layers=None,
            dropout=0.0,
            conv="GINConv",
            keys=["pathomics"],
            aggregation="CBM",
            **kwargs,
    ):
        super().__init__()
        if layers is None:
            layers = [6, 6]
        self.dropout = dropout
        self.embedding_dims = layers
        self.keys = keys
        self.aggregation = aggregation
        self.num_layers = len(self.embedding_dims)
        self.enc_convs = nn.ModuleList()
        self.enc_linears = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.dec_linears = nn.ModuleList()
        self.conv_name = conv
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                ReLU(),
                Dropout(self.dropout)
            )
        
        if len(keys) > 1:
            out_emb_dim = self.embedding_dims[0]
            self.enc_branches = nn.ModuleDict({k: create_block(dim_features[k], out_emb_dim) for k in keys})
            self.dec_branches = nn.ModuleDict({k: create_block(out_emb_dim, dim_features[k]) for k in keys})
            input_emb_dim = out_emb_dim
            out_emb_dim = self.embedding_dims[0]
        elif len(keys) == 1:
            input_emb_dim = dim_features[keys[0]]
            out_emb_dim = self.embedding_dims[0]
        else:
            raise NotImplementedError

        self.head = create_block(input_emb_dim, out_emb_dim)
        input_emb_dim = out_emb_dim

        if aggregation == "ABMIL":
            self.gate_nn = Attn_Net_Gated(
                L=input_emb_dim,
                D=256,
                dropout=0.25,
                n_classes=1
            )
        elif aggregation == "CBM":
            out_emb_dim = dim_concept
            self.concept_nn = ConceptScoreArch(
                dim_features=input_emb_dim,
                dim_target=out_emb_dim,
                layers=self.embedding_dims[1:],
                dropout=self.dropout,
                conv=self.conv_name
            )
            input_emb_dim = out_emb_dim
            self.gate_nn = Attn_Net_Gated(
                L=input_emb_dim,
                D=64,
                dropout=0.25,
                n_classes=dim_concept
            )
        else:
            raise NotImplementedError
        self.SumAggregation = SumAggregation()
        self.classifier = Linear(input_emb_dim, dim_target)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def forward(self, data):
        if len(self.keys) > 1:
            x_dict = data.x_dict
            edge_index_dict = data.edge_index_dict
            batch_dict = data.batch_dict
            enc_list = [x_dict[k] for k in self.keys]
            feature_list = [self.enc_branches[k](enc) for k, enc in zip(self.keys, enc_list)]
            feature = torch.concat(feature_list, dim=0)
            edge_index_list = [edge_index_dict[k, "to", k] for k in self.keys]
            # check = [[len(f), e.max().item()] for f, e in zip(feature_list, edge_index_list)]
            # print("Number of nodes and edge index: ", check)
            index_shift = 0
            for i in range(1, len(edge_index_list)): 
                index_shift += len(feature_list[i-1])
                edge_index_list[i] += index_shift
            edge_index = torch.concat(edge_index_list, dim=1)
            batch_list = [batch_dict[k] for k in self.keys]
            batch = torch.concat(batch_list, dim=0)
        else:
            feature, edge_index, batch = data.x, data.edge_index, data.batch

        feature = self.head(feature)
        if self.aggregation == "ABMIL":
            gate = self.gate_nn(feature)
        elif self.aggregation == "CBM":
            feature = self.concept_nn(feature, edge_index)
            gate = self.gate_nn(feature)
        gate = softmax(gate, index=batch, dim=-2)
        feature = self.SumAggregation(feature * gate, index=batch)
        if self.aggregation == "ABMIL":
            concept_logit = None
        elif self.aggregation == "CBM":
            concept_logit = feature
        output = self.classifier(feature)
        return output, concept_logit
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        model.train()
        optimizer.zero_grad()
        wsi_outputs, concept_logits = model(wsi_graphs)
        wsi_outputs = wsi_outputs.squeeze()
        if hasattr(wsi_graphs, "y_dict"):
            wsi_labels = wsi_graphs.y_dict[model.keys[0]].squeeze()
            concept_labels = wsi_graphs.concept_dict[model.keys[0]].squeeze()
        elif hasattr(wsi_graphs, "y"):
            wsi_labels = wsi_graphs.y.squeeze()
            concept_labels = wsi_graphs.concept.squeeze()
        else:
            raise AttributeError
        loss = loss(wsi_outputs, wsi_labels, concept_logits, concept_labels)
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

        model.eval()
        with torch.inference_mode():
            wsi_outputs, concept_logits = model(wsi_graphs)
        wsi_outputs = wsi_outputs.cpu().numpy()
        wsi_labels, concept_labels = None, None
        if hasattr(wsi_graphs, "y_dict"):
            wsi_labels = wsi_graphs.y_dict[model.keys[0]].cpu().numpy()
            concept_labels = wsi_graphs.concept_dict[model.keys[0]].cpu().numpy()
        elif hasattr(wsi_graphs, "y"):
            wsi_labels = wsi_graphs.y.cpu().numpy()
            concept_labels = wsi_graphs.concept.cpu().numpy()
        if concept_logits is not None:
            concept_logits = concept_logits.cpu().numpy()
            return [wsi_outputs, wsi_labels, concept_logits, concept_labels]
        else:
            return [wsi_outputs, wsi_labels]

class SubtypingGraphArch(nn.Module):
    """define Graph architecture for cancer subtyping
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
            "MLP": [None, 1],
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
                ReLU(),
                Dropout(self.dropout)
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        input_emb_dim = out_emb_dim
        out_emb_dim = self.embedding_dims[-1]

        # input_emb_dim = out_emb_dim
        # for out_emb_dim in self.embedding_dims[1:]:
        #     conv_class, alpha = conv_dict[self.conv_name]
        #     if self.conv_name in ["GINConv", "EdgeConv"]:
        #         block = create_block(alpha * input_emb_dim, out_emb_dim)
        #         subnet = conv_class(block, **kwargs)
        #         self.convs.append(subnet)
        #         self.linears.append(Linear(out_emb_dim, out_emb_dim))
        #     elif self.conv_name in ["GCNConv", "GATConv"]:
        #         subnet = conv_class(alpha * input_emb_dim, out_emb_dim)
        #         self.convs.append(subnet)
        #         self.linears.append(create_block(out_emb_dim, out_emb_dim))
        #     else:
        #         subnet = create_block(alpha * input_emb_dim, out_emb_dim)
        #         self.convs.append(subnet)
        #         self.linears.append(nn.Sequential())
                
        #     input_emb_dim = out_emb_dim
            
        self.gate_nn = Attn_Net_Gated(
            L=input_emb_dim,
            D=out_emb_dim,
            dropout=0.25,
            n_classes=1
        )
        self.global_attention = AttentionalAggregation(
            gate_nn=self.gate_nn,
            nn=None
        )
        self.classifier = Linear(input_emb_dim, dim_target)

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
        # for layer in range(1, self.num_layers):
        #     # feature = F.dropout(feature, p=self.dropout, training=self.training)
        #     if self.conv_name in ["MLP"]:
        #         feature = self.convs[layer - 1](feature)
        #     else:
        #         feature = self.convs[layer - 1](feature, edge_index)
        #     feature = self.linears[layer - 1](feature)

        feature = self.global_attention(feature, index=batch)
        output = self.classifier(feature)
        return output
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer, kl=None):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.train()
        optimizer.zero_grad()
        wsi_outputs = model(wsi_graphs)
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
        wsi_outputs = wsi_outputs.cpu().numpy()
        if wsi_graphs.y is not None:
            wsi_labels = wsi_graphs.y.cpu().numpy()
            return [wsi_outputs, wsi_labels]
        return [wsi_outputs]

class SubtypingBayesGraphArch(nn.Module):
    """define Graph architecture for cancer subtyping
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers=None,
            dropout=0.0,
            conv="GINConv",
            Bayes_std=0.1,
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
            "MLP": [None, 1],
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
                ReLU(),
                Dropout(self.dropout)
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        input_emb_dim = out_emb_dim
        out_emb_dim = self.embedding_dims[-1]

        # input_emb_dim = out_emb_dim
        # for out_emb_dim in self.embedding_dims[1:]:
        #     conv_class, alpha = conv_dict[self.conv_name]
        #     if self.conv_name in ["GINConv", "EdgeConv"]:
        #         block = create_block(alpha * input_emb_dim, out_emb_dim)
        #         subnet = conv_class(block, **kwargs)
        #         self.convs.append(subnet)
        #         self.linears.append(bnn.BayesLinear(0, Bayes_std, input_emb_dim, out_emb_dim))
        #     elif self.conv_name in ["GCNConv", "GATConv"]:
        #         subnet = conv_class(alpha * input_emb_dim, out_emb_dim)
        #         self.convs.append(subnet)
        #         self.linears.append(create_block(out_emb_dim, out_emb_dim))
        #     else:
        #         subnet = create_block(alpha * input_emb_dim, out_emb_dim)
        #         self.convs.append(subnet)
        #         self.linears.append(nn.Sequential())
                
        #     input_emb_dim = out_emb_dim
            
        self.gate_nn = Bayes_Attn_Net_Gated(
            L=input_emb_dim,
            D=out_emb_dim,
            dropout=0.25,
            n_classes=1,
            Bayes_std=Bayes_std
        )
        self.global_attention = AttentionalAggregation(
            gate_nn=self.gate_nn,
            nn=None
        )
        self.classifier = Linear(input_emb_dim, dim_target)

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
        # for layer in range(1, self.num_layers):
        #     # feature = F.dropout(feature, p=self.dropout, training=self.training)
        #     if self.conv_name in ["MLP"]:
        #         feature = self.convs[layer - 1](feature)
        #     else:
        #         feature = self.convs[layer - 1](feature, edge_index)
        #     feature = self.linears[layer - 1](feature)

        feature = self.global_attention(feature, index=batch)
        output = self.classifier(feature)
        return output
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer, kl):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.train()
        optimizer.zero_grad()
        wsi_outputs = model(wsi_graphs)
        wsi_outputs = wsi_outputs.squeeze()
        wsi_labels = wsi_graphs.y.squeeze()
        loss = loss(wsi_outputs, wsi_labels)
        kl_loss, kl_weight = kl["loss"](model)[0], kl["weight"]
        loss = loss + kl_weight * kl_loss
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
        wsi_outputs = wsi_outputs.cpu().numpy()
        if wsi_graphs.y is not None:
            wsi_labels = wsi_graphs.y.cpu().numpy()
            return [wsi_outputs, wsi_labels]
        return [wsi_outputs]

class SlideBayesGraphArch(nn.Module):
    """define SlideBayesGraph architecture
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
            "MLP": [None, 1],
            "GCNConv": [GCNConv, 1],
            "GATConv": [GATConv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                bnn.BayesLinear(0, 0.1, in_dims, out_dims),
                BatchNorm1d(out_dims),
                ReLU(),
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        self.tail = {
            "MLP": bnn.BayesLinear(0, 0.1, self.embedding_dims[-1], dim_target),
            "GCNConv": GCNConv(self.embedding_dims[-1], dim_target),
            "GATConv": GATConv(self.embedding_dims[-1], dim_target),
            "GINConv": bnn.BayesLinear(0, 0.1, self.embedding_dims[-1], dim_target),
            "EdgeConv": bnn.BayesLinear(0, 0.1, self.embedding_dims[-1], dim_target)
        }[self.conv_name]

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embedding_dims[1:]:
            conv_class, alpha = conv_dict[self.conv_name]
            if self.conv_name in ["GINConv", "EdgeConv"]:
                block = create_block(alpha * input_emb_dim, out_emb_dim)
                subnet = conv_class(block, **kwargs)
                self.convs.append(subnet)
                self.linears.append(bnn.BayesLinear(0, 0.1, out_emb_dim, out_emb_dim))
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
    def train_batch(model, batch_data, on_gpu, loss, kl, optimizer, probs=None, tau=1):
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
        kl_loss, kl_weight = kl["loss"](model)[0], kl["weight"]
        loss = loss + kl_weight * kl_loss
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = wsi_labels.cpu().numpy()
        return [loss, wsi_outputs, wsi_labels]
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu, mode):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.eval()
        with torch.inference_mode():
            wsi_outputs = model(wsi_graphs)
        wsi_outputs = wsi_outputs.squeeze().cpu().numpy()
        if wsi_graphs.y is not None:
            wsi_labels = wsi_graphs.y.squeeze().cpu().numpy()

            ## convert outputs and labels based given mode
            if mode in ["PN", "uPU", "nnPU"]:
                wsi_outputs = wsi_outputs
                wsi_labels = np.array(wsi_labels > 0, dtype=np.int32)
            elif mode == "PCE":
                postive = wsi_labels > 0
                wsi_outputs = wsi_outputs[postive]
                wsi_labels = wsi_labels[postive] - 1
            elif mode == "LHCE":
                low = wsi_labels == 1
                high = wsi_labels == 3
                low_outputs, low_labels = wsi_outputs[low], wsi_labels[low] - 1
                high_outputs, high_labels = wsi_outputs[high], wsi_labels[high] - 2
                wsi_outputs = np.concatenate([low_outputs, high_outputs], axis=0)
                wsi_labels = np.concatenate([low_labels, high_labels], axis=0)
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

        positive_risk = 0.8 * self.prior * torch.mean(y_positive[positive])
        negative_risk = 0.2 * torch.mean(y_unlabeled[unlabeled]) - 0.2 * self.prior * torch.mean(y_unlabeled[positive])

        if self.mode == "nnPU":
            if negative_risk < -self.beta:
                return -self.gamma * negative_risk
            else:
                return positive_risk + negative_risk
        elif self.mode == "uPU":
            return positive_risk + negative_risk
        elif self.mode == "PN":
            positive_risk = self.prior * torch.mean(y_positive[positive])
            negative_risk = (1 - self.prior) * torch.mean(y_unlabeled[unlabeled])
            return positive_risk + negative_risk
        else:
            raise ValueError(f"{self.mode} is not supported!")
        
class PCELoss(nn.Module):
    """wrapper of loss function for positive samples"""

    def __init__(self):
        super(PCELoss, self).__init__()
        self.unlabeled = 0
        self.CrossEntropy = nn.CrossEntropyLoss()
    
    def forward(self, input, target):   
        positive = target > self.unlabeled
        ## loss only for postive samples
        input = input[positive]
        target = target[positive] - 1
        return self.CrossEntropy(input, target)
    

class LHCELoss(nn.Module):
    """wrapper of loss function for low-grade and high-grade learning"""

    def __init__(self):
        super(LHCELoss, self).__init__()
        self.lowgrade = 1
        self.highgrade = 3
        self.CrossEntropy = nn.CrossEntropyLoss()
    
    def forward(self, input, target):   
        lowgrade = target == self.lowgrade
        highgrade = target == self.highgrade
        ## loss only for postive samples
        input_lowgrade = input[lowgrade]
        input_highgrade = input[highgrade]
        target_lowgrade = target[lowgrade] - 1
        target_highgrade = target[highgrade] - 2
        input = torch.concat([input_lowgrade, input_highgrade])
        target = torch.concat([target_lowgrade, target_highgrade])
        return self.CrossEntropy(input, target)
    
class CoxSurvLoss(object):
    def __call__(self, hazards, time, c, VIparas, 
                 mu0=0, lambda0=1, alpha0=2, beta0=1e-8, tau_ae=1e-4, tau_kl=1e-4,
                 **kwargs
        ):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(time)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = time[j] >= time[i]

        # c = torch.tensor(c).to(hazards.device)
        R_mat = torch.tensor(R_mat).to(hazards.device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * c)
        if VIparas is not None:
            loc, logvar, enc_list, dec_list = VIparas
            mu_upsilon = (2*alpha0 + 1) / (2*beta0 + lambda0*(loc - mu0)**2 + lambda0*torch.exp(logvar))
            mu_upsilon = mu_upsilon.clone().detach()
            loss_kl = 0.5*torch.mean(lambda0*mu_upsilon*((loc - mu0)**2 + torch.exp(logvar)) - logvar)
            loss_ae = sum([F.mse_loss(enc, dec) for enc, dec in zip(enc_list, dec_list)])
            print(loss_cox.item(), loss_ae.item(), loss_kl.item())
            loss_cox = loss_cox + tau_ae*loss_ae + tau_kl*loss_kl
        # print(loss_cox)
        # print(R_mat)
        return loss_cox
    
class CoxSurvConceptLoss(object):
    def __init__(self, tau=1.0, concept_weight=None):
        self.tau = tau
        self.concept_weight = concept_weight
        
    def __call__(self, hazards, labels, 
                 concept_logits, concept_labels,
                 **kwargs
        ):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        time, c = labels[:, 0], labels[:, 1]
        current_batch_len = len(time)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = time[j] >= time[i]

        # c = torch.tensor(c).to(hazards.device)
        R_mat = torch.tensor(R_mat).to(hazards.device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * c)
        if concept_logits is not None:
            concept_probs = torch.sigmoid(concept_logits)
            if self.concept_weight is not None:
                weight = torch.tensor(self.concept_weight).to(concept_labels.device)
                loss_cbm = F.binary_cross_entropy(concept_probs, concept_labels, weight)
            else:
                loss_cbm = F.binary_cross_entropy(concept_probs, concept_labels)
            loss_cox = loss_cox + self.tau * loss_cbm
        # print(loss_cox)
        # print(R_mat)
        return loss_cox
    
def update_loss(mode, prior=0.5):
    loss = {
        "CE": torch.nn.CrossEntropyLoss(),
        "PN": PULoss(prior=prior, mode="PN"),
        "uPU": PULoss(prior=prior, mode="uPU"),
        "nnPU": PULoss(prior=prior, mode="nnPU"),
        "PCE": PCELoss(),
        "LHCE": LHCELoss()
    }[mode]
    return loss
