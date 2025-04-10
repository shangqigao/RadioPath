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

    
class ConceptGraphDataset(Dataset):
    """loading graph data for concept learning
    """
    def __init__(self, info_list, mode="train", preproc=None, data_types=["pathomics"], use_histopath=False):
        super().__init__()
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
        self.data_types = data_types
        self.use_histopath = use_histopath
    
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

            if self.use_histopath:
                histopath = np.load(f"{graph_path[key]}".replace(".json", ".label.npy"))
                graph_dict["x"] = np.concatenate([graph_dict["x"], histopath], axis=1)

            graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
            if any(v in self.mode for v in ["train", "valid"]):
                graph_dict.update({"concept": concept, "y": label})
            
            graph_dict = {k: v.type(torch.float32) for k, v in graph_dict.items()}
            graph_dict.update({"edge_index": graph_dict["edge_index"].type(torch.int64)})
            graph = Data(**graph_dict)
            del graph_dict
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
                del edge_dict
                del subgraph_dict
            
            graph = HeteroData(graph_dict)
        return graph
    
    def len(self):
        return len(self.info_list)

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
            # self.concept_mlp = Linear(input_emb_dim, dim_concept)
            # input_emb_dim = dim_concept
        else:
            raise NotImplementedError
        self.SumAggregation = SumAggregation()
        self.classifier = Linear(input_emb_dim, dim_target)

        self.aux_model = {}

    def save(self, path, aux_path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        joblib.dump(self.aux_model, aux_path)

    def load(self, path, aux_path, on_gpu=False):
        if on_gpu:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
        self.aux_model = joblib.load(aux_path)

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
            # feature = self.concept_mlp(feature)
            concept_logit = feature
        output = self.classifier(feature)
        return output, concept_logit, gate, feature
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        model.train()
        optimizer.zero_grad()
        wsi_outputs, concept_logits, _, _ = model(wsi_graphs)
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
            wsi_outputs, concept_logits, attention, feature = model(wsi_graphs)
        wsi_outputs = wsi_outputs.cpu().numpy()
        attention = attention.cpu().numpy()
        feature = feature.cpu().numpy()
        wsi_labels, concept_labels = None, None
        if hasattr(wsi_graphs, "y_dict"):
            if wsi_graphs.y_dict is not None:
                wsi_labels = wsi_graphs.y_dict[model.keys[0]].cpu().numpy()
                concept_labels = wsi_graphs.concept_dict[model.keys[0]].cpu().numpy()
        elif hasattr(wsi_graphs, "y"):
            if wsi_graphs.y is not None:
                wsi_labels = wsi_graphs.y.cpu().numpy()
                concept_labels = wsi_graphs.concept.cpu().numpy()
                
        if concept_logits is not None:
            concept_logits = concept_logits.cpu().numpy()
            if wsi_labels is not None:
                return [wsi_outputs, wsi_labels, concept_logits, concept_labels]
            else:
                # return [wsi_outputs, concept_logits, attention]
                return [wsi_outputs, concept_logits]
        else:
            if wsi_labels is not None:
                return [wsi_outputs, wsi_labels]
            else:
                # return [wsi_outputs, feature, attention]
                return [wsi_outputs, feature]

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
        return loss_cbm